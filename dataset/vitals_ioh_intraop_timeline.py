"""
脚本功能：
- 读取 `data_process.vitals_ioh_timeline` 表中的 IOH 波形数据（已有的时间轴数据）。
- 读取 `data_process.cases_filter` 表中的手术时间窗口 (`anestart`, `aneend`)。
- 解析 `vitals_ioh_timeline` 中的 `chart_times` (JSONB数组) 和 6个波形字段 (JSONB数组)。
- 补全或截取时间轴，使其完全覆盖 `anestart` 到 `aneend` 范围（每2秒一个点）。
- 写入目标表 `data_process.vitals_ioh_intraop_timeline`。

补全逻辑：
- 如果 `anestart` 是奇数，从下一个偶数开始。
- 目标时间轴范围：从 `adjusted_anestart` 到 `aneend`，步长为 2。
- 对于目标时间轴上的每个点：
    - 如果在原 `chart_times` 中存在，取对应的 `value`。
    - 如果不存在，`value` 补 `null`。
- 最终 `chart_times` 和所有波形 `values` 长度必须一致。

涉及的6个波形字段：
- solar8000/art_dbp
- solar8000/art_mbp
- solar8000/art_sbp
- solar8000/etco2
- solar8000/vent_mawp
- solar8000/hr

输入参数：
- 连接信息：`--src-host/--src-port/--src-dbname/--src-user/--src-password`
- 源/目标表：`--cases-schema --cases-table` (默认 data_process.cases_filter)
            `--vitals-schema --vitals-table` (默认 data_process.vitals_ioh_timeline)
            `--dest-schema --dest-table` (默认 data_process.vitals_ioh_intraop_timeline)
- `--analyze` 对目标表执行 `ANALYZE`
"""
import argparse
import psycopg2
import json
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-host", default="127.0.0.1")
    parser.add_argument("--src-port", type=int, default=5434)
    parser.add_argument("--src-dbname", default="postgres")
    parser.add_argument("--src-user", default="postgres")
    parser.add_argument("--src-password", default="123456")
    
    parser.add_argument("--cases-schema", default="data_process")
    parser.add_argument("--cases-table", default="cases_filter")
    
    parser.add_argument("--vitals-schema", default="data_process")
    parser.add_argument("--vitals-table", default="vitals_ioh_timeline")
    
    parser.add_argument("--dest-schema", default="data_process")
    parser.add_argument("--dest-table", default="vitals_ioh_intraop_timeline")
    
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    # Column names (lowercased as per import_ioh_data.py)
    waveform_cols = [
        "solar8000/art_dbp",
        "solar8000/art_mbp",
        "solar8000/art_sbp",
        "solar8000/etco2",
        "solar8000/vent_mawp",
        "solar8000/hr"
    ]

    conn = psycopg2.connect(
        host=args.src_host, 
        port=args.src_port, 
        dbname=args.src_dbname, 
        user=args.src_user, 
        password=args.src_password
    )
    conn.autocommit = True
    
    try:
        with conn.cursor() as cur:
            cases_tbl = f"{args.cases_schema}.{args.cases_table}"
            vitals_tbl = f"{args.vitals_schema}.{args.vitals_table}"
            dest_tbl = f"{args.dest_schema}.{args.dest_table}"
            
            # Create Schema
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {args.dest_schema}")
            
            # Create Dest Table
            # Note: We need to quote column names because of special characters
            cols_def = ", ".join([f'"{col}" JSONB' for col in waveform_cols])
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {dest_tbl} (
                    caseid INTEGER,
                    subjectid INTEGER,
                    anestart INTEGER,
                    aneend INTEGER,
                    chart_times JSONB,
                    {cols_def}
                )
            """)
            
            # Create Unique Index
            cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {args.dest_table}_unq ON {dest_tbl} (caseid)")

            print("Fetching cases...")
            # Get cases with valid anestart/aneend
            cur.execute(f"""
                SELECT c.caseid, c.subjectid, c.anestart, c.aneend
                FROM {cases_tbl} c
                WHERE c.anestart IS NOT NULL AND c.aneend IS NOT NULL
                ORDER BY c.caseid
            """)
            cases = cur.fetchall()
            
            total_cases = len(cases)
            processed = 0
            inserted = 0
            
            for c in cases:
                caseid = c[0]
                subjectid = c[1]
                anestart = int(float(c[2]))
                aneend = int(float(c[3]))
                
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed}/{total_cases} cases...")
                
                # Check if exists in dest table
                cur.execute(f"SELECT 1 FROM {dest_tbl} WHERE caseid = %s", (caseid,))
                if cur.fetchone():
                    continue

                # Fetch existing timeline data
                # Select chart_times and all waveform columns
                select_cols = ", ".join([f'"{col}"' for col in waveform_cols])
                # Use format to inject table name and columns safely
                # Note: vitals_tbl is constructed from trusted args or defaults
                query = f'SELECT chart_times, {select_cols} FROM {vitals_tbl} WHERE caseid = %s'
                cur.execute(query, (caseid,))
                row = cur.fetchone()
                
                # If row is None or chart_times is None, treat as empty
                if not row or row[0] is None:
                    src_chart_times = []
                    src_values_list = [[] for _ in waveform_cols]
                else:
                    src_chart_times = row[0]
                    # row[1:] are the waveform value lists
                    src_values_list = []
                    for val in row[1:]:
                        src_values_list.append(val if val is not None else [])
                
                # Create lookup maps for each waveform
                # data_maps[i] corresponds to waveform_cols[i]
                data_maps = [{} for _ in waveform_cols]
                
                # Assuming all src_values lists are same length as src_chart_times
                # Check lengths to be safe
                for i, src_values in enumerate(src_values_list):
                    if len(src_chart_times) == len(src_values):
                        for t, v in zip(src_chart_times, src_values):
                            if t is not None:
                                data_maps[i][int(t)] = v
                
                # Determine target timeline start (Align to Even)
                if anestart % 2 != 0:
                    target_start = anestart + 1
                else:
                    target_start = anestart
                
                target_end = aneend
                
                valid_times = []
                # valid_values_list[i] will hold values for waveform_cols[i]
                valid_values_list = [[] for _ in waveform_cols]
                
                # Generate full timeline
                if target_start <= target_end:
                    for t in range(target_start, target_end + 1, 2):
                        valid_times.append(t)
                        
                        for i in range(len(waveform_cols)):
                            val = data_maps[i].get(t)
                            # Sanitize NaN to None
                            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                                val = None
                            valid_values_list[i].append(val)
                
                if valid_times:
                    # Prepare INSERT
                    insert_cols = ["caseid", "subjectid", "anestart", "aneend", "chart_times"] + [f'"{col}"' for col in waveform_cols]
                    placeholders = ["%s"] * len(insert_cols)
                    
                    cols_str = ", ".join(insert_cols)
                    placeholders_str = ", ".join(placeholders)
                    
                    values_to_insert = [caseid, subjectid, anestart, aneend, json.dumps(valid_times)]
                    for valid_values in valid_values_list:
                        values_to_insert.append(json.dumps(valid_values))
                    
                    insert_query = f'INSERT INTO {dest_tbl} ({cols_str}) VALUES ({placeholders_str})'
                    cur.execute(insert_query, tuple(values_to_insert))
                    inserted += 1
            
            print(f"Completed. Processed: {processed}, Inserted: {inserted}")
            
            if args.analyze:
                cur.execute(f"ANALYZE {dest_tbl}")
                
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    # python vitals_ioh_intraop_timeline.py --src-host 172.16.60.23 --src-port 5434 --src-dbname postgres --src-user postgres --src-password 123456 --cases-schema data_process --cases-table cases_filter --vitals-schema data_process --vitals-table vitals_ioh_timeline --dest-schema data_process --dest-table vitals_ioh_intraop_timeline --analyze
    main()
