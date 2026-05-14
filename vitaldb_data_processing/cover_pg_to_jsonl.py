import json
import psycopg2
from psycopg2 import OperationalError
from tqdm import tqdm
from decimal import Decimal


def create_connection():
    """创建数据库连接"""
    try:
        conn = psycopg2.connect(
            host="172.16.60.23",
            port="5434",
            database="postgres",
            user="postgres",
            password="123456"
        )
        print("✅ 数据库连接成功")
        return conn
    except OperationalError as e:
        print(f"❌ 数据库连接失败: {e}")
        return None


def process_table_to_jsonl(connection, output_file_path):
    """从表中查询数据并转换为JSONL文件"""
    query = """
    SELECT caseid, stime, "time", age, sex, bmi, asa,
           solar8000_art_dbp, solar8000_art_mbp, solar8000_art_sbp,
           solar8000_etco2, solar8000_vent_mawp, solar8000_hr
    FROM data_process.ioh_dataset
    """
    
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        print(f"📊 查询到 {len(rows)} 条记录")
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for row in tqdm(rows, desc="📝 转换为JSONL"):
                try:
                    record = {
                        "id": row[0],
                        "stime": row[1],
                        "time": row[2] if row[2] is not None else "",
                        "age": float(row[3]) if row[3] is not None else None,
                        "sex": int(row[4]) if row[4] is not None else None,
                        "bmi": float(row[5]) if row[5] is not None else None,
                        "asa": int(row[6]) if (row[6] is not None and not row[6].is_nan()) else None,
                        "ART_DBP": row[7] if row[7] is not None else [],
                        "ART_MBP": row[8] if row[8] is not None else [],
                        "ART_SBP": row[9] if row[9] is not None else [],
                        "ETCO2": row[10] if row[10] is not None else [],
                        "VENT_MAWP": row[11] if row[11] is not None else [],
                        "HR": row[12] if row[12] is not None else []
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"⚠️ 记录转换失败 (ID: {row[0]}): {e}")
                    continue
        
        print(f"✅ 转换完成，输出文件: {output_file_path}")
    
    except Exception as e:
        print(f"❌ 查询或转换失败: {e}")
    finally:
        cursor.close()


if __name__ == "__main__":
    
    # nohup python VitalDB_20251220/cover_pg_to_jsonl.py > cover_pg_to_jsonl.log 2>&1 &
    
    output_file = "VitalDB_20251220/ioh_dataset_vitaldb_1810.jsonl"
    conn = create_connection()
    if conn:
        process_table_to_jsonl(conn, output_file)
        conn.close()
        print("🔒 数据库连接已关闭")
