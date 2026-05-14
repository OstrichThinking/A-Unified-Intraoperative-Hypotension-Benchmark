import os
import sys
from turtle import pd
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, Json
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# ========== 批处理参数 ==========
BATCH_SIZE = 100  # 每批读取条数，可调

# ========== 批处理参数 ==========

def detect_ioh(map_json, time_json, anes_start, stime, threshold=65.0, duration=60):
    if not map_json or not time_json or len(map_json) != len(time_json):
        return 0, None
    
    ioh_occurred = 0
    onset_time = None
    current_streak = 0
    
    for i in range(len(map_json)):
        if map_json[i] is None:
            current_streak = 0
            continue
        if map_json[i] < threshold:
            current_streak += 1
            if current_streak >= int(duration // stime) and not ioh_occurred:
                ioh_occurred = 1
                # The onset starts at the first point of the streak
                onset_time = time_json[i - 1]  # i-1 is the start of the streak
        else:
            current_streak = 0
    
    if ioh_occurred:
        # Calculate offset in minutes from anes_start
        onset_offset = (onset_time - anes_start) / 60.0
        return 1, onset_offset
    else:
        return 0, None

def valid_filter(caseid, ART_DBP, ART_MBP, ART_SBP, ETCO2, VENT_MAWP, HR):
    is_valid = True

    # 将所有指标放在一起统一处理
    signals = [ART_DBP, ART_MBP, ART_SBP, ETCO2, VENT_MAWP, HR]
    signals_name = ['ART_DBP', 'ART_MBP', 'ART_SBP', 'ETCO2', 'VENT_MAWP', 'HR']

    for sig, sig_name in zip(signals, signals_name):
        if sig is None:
            # 如果整个信号是 None，直接判无效
            is_valid = False
            break
        
        n = len(sig)
        if n == 0:
            is_valid = False
            break
        
        none_count = sum(v is None for v in sig)

        # 若 None 比例 > 20%
        if none_count > 0.2 * n:
            is_valid = False
            print(f"  caseid{caseid} - {sig_name} 缺失比例过高，跳过")
            break

    return is_valid

def range_valid_filter(ART_DBP, ART_MBP, ART_SBP, ETCO2, VENT_MAWP, HR):
    
    def process(data, min_v=None, max_v=None):
        """通用异常值处理函数"""
        processed = []
        for d in data:
            # 判断范围
            if d is not None:
                if (min_v is not None and d < min_v) or (max_v is not None and d > max_v):
                    d = None
            processed.append(d)
        return processed

    # ART_DBP 有创舒张压
    ART_DBP_processed = process(ART_DBP, 0, 165)

    # ART_SBP 有创收缩压
    ART_SBP_processed = process(ART_SBP, 45, 270)

    # ART_MBP 有创平均动脉压
    ART_MBP_processed = process(ART_MBP, 40, 150)

    # ETCO2 呼气末二氧化碳
    ETCO2_processed = process(ETCO2, 17.5, 67.5)

    # VENT_MAWP 气道压，无硬性下限，仅上限
    VENT_MAWP_processed = process(VENT_MAWP, 0, 52.5)

    # HR 心率
    HR_processed = process(HR, 30, 150)

    return (ART_DBP_processed, ART_MBP_processed, ART_SBP_processed, ETCO2_processed, VENT_MAWP_processed,
            HR_processed)

# ========== 主迁移函数 ==========
def ioh_dataset_1_valid(conn_read, conn_write):
    
    # 总行数用于进度展示
    cur_count = conn_read.cursor()
    cur_count.execute("""SELECT COUNT(*)
        FROM data_process.vitals_ioh_intraop_timeline v
        JOIN data_process.cases_filter cf  ON v.caseid = cf.caseid
        WHERE cf.opname NOT LIKE '%transplant%'
        AND cf.emop = 0""")
    total_rows = cur_count.fetchone()[0]
    cur_count.close()

    cur_in = conn_read.cursor(name='ioh_dataset')
    cur_out = conn_write.cursor()
    
    # 排除移植手术病例
    sql = """SELECT v.caseid, v.subjectid, v.anestart, v.aneend, v.chart_times, 
        v."solar8000/art_dbp", v."solar8000/art_mbp", v."solar8000/art_sbp", v."solar8000/etco2", v."solar8000/vent_mawp", v."solar8000/hr",
        cf.age, cf.sex, cf.bmi, cf.asa
        FROM data_process.vitals_ioh_intraop_timeline v
        JOIN data_process.cases_filter cf ON v.caseid = cf.caseid
        WHERE cf.opname NOT LIKE '%transplant%'
        AND cf.emop = 0"""
    
    cur_in.itersize = BATCH_SIZE
    cur_in.execute(sql)
    
    total_inserted = 0
    pbar = tqdm(total=total_rows, desc="Process 1: Valid Filtering")
    while True:
        rows = cur_in.fetchmany(BATCH_SIZE)
        if not rows:
            break

        pbar.update(len(rows))
        data_to_insert = []
        for row in rows:
            (caseid, subjectid, anestart, aneend, chart_times, 
             Solar8000_ART_DBP, Solar8000_ART_MBP, Solar8000_ART_SBP, Solar8000_ETCO2, Solar8000_VENT_MAWP, Solar8000_HR,
             age, sex, bmi, asa) = row
            
            sex = 1 if sex == 'F' else 0
            
            if (anestart is None) or (aneend is None) or ((aneend - anestart) < 3600) or (anestart >= aneend):
                if anestart is None:
                    print(f"  - 麻醉开始时间缺失，跳过 (caseid={caseid})")
                if aneend is None:
                    print(f"  - 麻醉结束时间缺失，跳过 (caseid={caseid})")
                if (aneend - anestart) < 3600:
                    print(f"  - 麻醉时长小于1小时，跳过 (caseid={caseid})")
                if anestart >= aneend:
                    print(f"  - 麻醉开始时间大于麻醉结束时间异常，跳过 (caseid={caseid})")
                continue

            try:
                Solar8000_ART_DBP, Solar8000_ART_MBP, Solar8000_ART_SBP, Solar8000_ETCO2, Solar8000_VENT_MAWP, Solar8000_HR = range_valid_filter(
                    Solar8000_ART_DBP, Solar8000_ART_MBP, Solar8000_ART_SBP, Solar8000_ETCO2, Solar8000_VENT_MAWP, Solar8000_HR)
                
                ioh_label, ioh_time_min = detect_ioh(Solar8000_ART_MBP, chart_times, anestart, stime=2, threshold=65.0, duration=60)
                
                # 任意波形缺失 20% 判无效
                is_valid = valid_filter(caseid,
                    Solar8000_ART_DBP, Solar8000_ART_MBP, Solar8000_ART_SBP, Solar8000_ETCO2, Solar8000_VENT_MAWP, Solar8000_HR)
            
            except Exception as e:
                print(f"  - 数据处理错误，跳过 (caseid={caseid}): {e}")
                continue
            
            if is_valid:
                data_to_insert.append((
                    caseid, subjectid, 2, anestart, aneend, 
                    Json(chart_times), age, sex, bmi, asa,
                    Json(Solar8000_ART_DBP), Json(Solar8000_ART_MBP), Json(Solar8000_ART_SBP), Json(Solar8000_ETCO2), Json(Solar8000_VENT_MAWP), Json(Solar8000_HR),
                    ioh_label, ioh_time_min
                ))
            
        # 批量插入（空批次跳过）
        if data_to_insert:
            execute_values(cur_out, """
                    INSERT INTO data_process.ioh_dataset (
                    caseid, subjectid, stime, anes_start, anes_end, 
                    "time", age, sex, bmi, asa, 
                    solar8000_art_dbp, solar8000_art_mbp, solar8000_art_sbp, solar8000_etco2, solar8000_vent_mawp, solar8000_hr, 
                    ioh_label, ioh_time_min
                ) VALUES %s
            """, data_to_insert)
            
            total_inserted += len(data_to_insert)
            conn_write.commit()
            pbar.set_postfix(inserted=total_inserted)

    pbar.close()
    cur_in.close()
    cur_out.close()
    conn_read.close()
    conn_write.close()
    print(f"\n🎉 进程1完成，共插入 {total_inserted} 条记录。")


if __name__ == "__main__":

    # nohup python VitalDB_20251220/ioh_dataset.py > ioh_dataset.log 2>&1 &
    
    DB_CONFIG = {
        "host": "172.16.60.23",
        "port": 5434,
        "database": "postgres",
        "user": "postgres",
        "password": "123456"
    }
    conn_read = psycopg2.connect(**DB_CONFIG)
    conn_write = psycopg2.connect(**DB_CONFIG)
    ioh_dataset_1_valid(conn_read, conn_write)
    
    
    
    
