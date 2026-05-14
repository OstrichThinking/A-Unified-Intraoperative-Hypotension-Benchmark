"""
脚本功能：
- 读取 `output/vitals_ioh_timeline.jsonl` 文件
- 自动检测除 caseid, anestart, aneend, chart_times 之外的波形字段（预期有6个）
- 将波形数据作为 JSONB 字段导入 PostgreSQL 数据库表
- 字段映射：
  - `caseid`: caseid
  - `anestart`: anestart
  - `aneend`: aneend
  - `chart_times`: chart_times (JSONB)
  - [其他检测到的字段]: [字段名] (JSONB)

输入参数：
- 连接信息：`--src-host/--src-port/--src-dbname/--src-user/--src-password`
- 目标表：`--dest-schema --dest-table` (默认 data_process.vitals_ioh_timeline)
- 输入文件：`--input-file` (默认 output/vitals_ioh_timeline.jsonl)
- `--analyze` 对目标表执行 `ANALYZE`

示例输出：
{'dest': {'host': '172.16.60.23', 'port': 5434, 'dbname': 'postgres', 'schema': 'data_process', 'table': 'vitals_ioh_timeline'}, 'total_read': 3329, 'write': True, 'analyze': True, 'dest_count': 3329, 'dynamic_columns': ['solar8000/art_dbp', 'solar8000/art_mbp', 'solar8000/art_sbp', 'solar8000/etco2', 'solar8000/vent_mawp', 'solar8000/hr']}
"""
import argparse
import json
import psycopg2
from psycopg2 import extras
import os
import math
from tqdm import tqdm

def clean_float(text):
    try:
        val = float(text)
        if math.isinf(val) or math.isnan(val):
            return None
        return val
    except ValueError:
        return None

def repair_json(line, parse_constant=None, parse_float=None):
    # Remove null bytes and strip
    clean_line = line.replace('\x00', '').strip()
    
    try:
        return json.loads(clean_line, parse_constant=parse_constant, parse_float=parse_float)
    except json.JSONDecodeError:
        pass
        
    # Attempt repair by finding the last valid structural point
    last_comma = clean_line.rfind(',')
    last_open_sq = clean_line.rfind('[')
    last_open_cur = clean_line.rfind('{')
    
    # Find the rightmost delimiter
    # Note: We prioritize preserving structure. 
    # If the last thing is a comma, we cut before it.
    # If the last thing is an opener, we keep it.
    
    cut_pos = -1
    keep_delimiter = False
    
    if last_comma > last_open_sq and last_comma > last_open_cur:
        cut_pos = last_comma
        keep_delimiter = False
    else:
        cut_pos = max(last_open_sq, last_open_cur)
        keep_delimiter = True
        
    if cut_pos == -1:
         # If no structure found, maybe it's a primitive or broken
         raise ValueError("Repair failed: No delimiters found")
         
    if keep_delimiter:
        truncated = clean_line[:cut_pos+1]
    else:
        truncated = clean_line[:cut_pos]
        
    # Calculate closing brackets needed
    stack = []
    for char in truncated:
        if char == '{': stack.append('}')
        elif char == '[': stack.append(']')
        elif char == '}' or char == ']':
            if stack: stack.pop()
            
    repaired = truncated + "".join(reversed(stack))
    
    try:
        return json.loads(repaired, parse_constant=parse_constant, parse_float=parse_float)
    except Exception as e:
        raise ValueError(f"Repair failed after truncation: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-host", default="127.0.0.1")
    parser.add_argument("--src-port", type=int, default=5434)
    parser.add_argument("--src-dbname", default="postgres")
    parser.add_argument("--src-user", default="postgres")
    parser.add_argument("--src-password", default="123456")
    
    parser.add_argument("--dest-schema", default="data_process")
    parser.add_argument("--dest-table", default="vitals_ioh_timeline")
    # 注意：这里修改为正确的默认路径，假设脚本在 data_process/ioh 目录下运行，或者根据绝对路径
    # 用户的 LS 显示 output 在 data_process/ioh/output
    parser.add_argument("--input-file", default="output/vitals_ioh_timeline.jsonl")
    
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    # Database connection
    conn = psycopg2.connect(
        host=args.src_host, 
        port=args.src_port, 
        dbname=args.src_dbname, 
        user=args.src_user, 
        password=args.src_password
    )
    conn.autocommit = False
    
    # Error log file
    error_log_file = "./output/import_errors.log"

    try:
        with conn.cursor() as cur:
            # Check input file
            if not os.path.exists(args.input_file):
                # Try absolute path based on known structure if relative fails
                base_dir = os.path.dirname(os.path.abspath(__file__))
                alt_path = os.path.join(base_dir, args.input_file)
                if os.path.exists(alt_path):
                    args.input_file = alt_path
                else:
                    print(f"Warning: Input file not found: {args.input_file}")
                    raise FileNotFoundError(f"Input file not found: {args.input_file}")

            print(f"Reading schema from {args.input_file}...")
            # Pre-read first line to detect columns
            dynamic_keys = []     # Lowercase keys for database columns
            original_keys = []    # Original keys for data extraction
            
            with open(args.input_file, 'r', encoding='utf-8') as f:
                line = f.readline()
                if line:
                    try:
                        # parse_constant=lambda x: None handles NaN, Infinity, -Infinity literals in JSON
                        record = json.loads(line, parse_constant=lambda x: None, parse_float=clean_float)
                        all_keys = list(record.keys())
                        fixed_keys = {'caseid', 'anestart', 'aneend', 'chart_times'}
                        original_keys = [k for k in all_keys if k not in fixed_keys]
                        dynamic_keys = [k.lower() for k in original_keys]
                        
                        # Check for collisions
                        if len(set(dynamic_keys)) != len(dynamic_keys):
                            raise ValueError(f"Column name collision after lowercasing: {original_keys} -> {dynamic_keys}")
                            
                        print(f"Detected dynamic waveform columns (original): {original_keys}")
                        print(f"Mapped to lowercase columns: {dynamic_keys}")
                    except json.JSONDecodeError as e:
                        print(f"Error reading schema from first line: {e}")
                        raise
                else:
                    raise ValueError("Input file is empty")

            print(f"Counting lines in {args.input_file}...")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                 total_lines = sum(1 for _ in f)

            print(f"Reading data from {args.input_file}...")
            data = []
            error_count = 0
            
            with open(error_log_file, 'w', encoding='utf-8') as err_f:
                err_f.write(f"Error log for {args.input_file}\n")
                
                with open(args.input_file, 'r', encoding='utf-8') as f:
                    for i, line in tqdm(enumerate(f), total=total_lines, desc="Processing"):
                        if not line.strip():
                            continue
                        try:
                            # Use parse_constant to handle NaN/Infinity literals -> None
                            # Use parse_float to handle numeric overflow (inf) or other float issues -> None
                            record = json.loads(line, parse_constant=lambda x: None, parse_float=clean_float)
                            
                            # Extract fields
                            caseid = int(record.get('caseid'))
                            anestart = int(record.get('anestart')) if record.get('anestart') is not None else None
                            aneend = int(record.get('aneend')) if record.get('aneend') is not None else None
                            
                            chart_times = record.get('chart_times')
                            if chart_times is not None:
                                chart_times_json = json.dumps(chart_times)
                            else:
                                chart_times_json = None
                            
                            row = [caseid, anestart, aneend, chart_times_json]
                            
                            # Extract dynamic fields using original keys
                            for key in original_keys:
                                val = record.get(key)
                                if val is not None:
                                    val_json = json.dumps(val)
                                else:
                                    val_json = None
                                row.append(val_json)
                            
                            data.append(tuple(row))
                        except (json.JSONDecodeError, ValueError, TypeError) as e:
                            error_count += 1
                            err_msg = f"Error decoding line {i+1}: {e}"
                            err_f.write(err_msg + "\n")
                            continue


            total = len(data)
            print(f"Loaded {total} records from file. (Skipped {error_count} error lines)")

            if not args.no_write:
                ds = args.dest_schema
                dt = args.dest_table
                
                # Create schema
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {ds}")
                
                # Create table
                # Base columns
                create_sql = f"""
                    CREATE TABLE IF NOT EXISTS {ds}.{dt} (
                        caseid INTEGER PRIMARY KEY, 
                        anestart INTEGER,
                        aneend INTEGER,
                        chart_times JSONB
                """
                # Dynamic columns
                for key in dynamic_keys:
                    create_sql += f', "{key}" JSONB'
                create_sql += ")"
                
                cur.execute(create_sql)
                
                print(f"Inserting into {ds}.{dt}...")
                
                # Construct INSERT query
                cols = ["caseid", "anestart", "aneend", "chart_times"] + [f'"{k}"' for k in dynamic_keys]
                placeholders = ["%s"] * len(cols)
                cols_str = ", ".join(cols)
                placeholders_str = ", ".join(placeholders)
                
                update_set = [f"anestart = EXCLUDED.anestart", f"aneend = EXCLUDED.aneend", f"chart_times = EXCLUDED.chart_times"]
                for key in dynamic_keys:
                    update_set.append(f'"{key}" = EXCLUDED."{key}"')
                update_str = ", ".join(update_set)
                
                insert_sql = f"""
                    INSERT INTO {ds}.{dt} ({cols_str}) 
                    VALUES ({placeholders_str}) 
                    ON CONFLICT (caseid) DO UPDATE 
                    SET {update_str}
                """
                
                extras.execute_batch(
                    cur,
                    insert_sql,
                    data,
                    page_size=100
                )
                conn.commit()
                print("Data insertion completed.")

            # Analyze
            dest_count = None
            if not args.no_write:
                cur.execute("SELECT to_regclass(%s)", (f"{ds}.{dt}",))
                reg = cur.fetchone()[0]
                if reg:
                    if args.analyze:
                        cur.execute(f"ANALYZE {ds}.{dt}")
                    cur.execute(f"SELECT COUNT(*) FROM {ds}.{dt}")
                    dest_count = cur.fetchone()[0]

            print({
                "dest": {"host": args.src_host, "port": args.src_port, "dbname": args.src_dbname, "schema": args.dest_schema, "table": args.dest_table},
                "total_read": total,
                "write": not args.no_write,
                "analyze": args.analyze,
                "dest_count": dest_count,
                "dynamic_columns": dynamic_keys
            })

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    # python import_ioh_data.py --src-host 172.16.60.23 --src-port 5434 --src-dbname postgres --src-user postgres --src-password 123456 --dest-schema data_process --dest-table vitals_ioh_timeline --analyze
    main()
