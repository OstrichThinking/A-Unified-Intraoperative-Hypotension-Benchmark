import os
import numpy as np
import pandas as pd
import json
import math

base_dir = "/home/share/datasets/ioh/VitalDB/processed/timeseries_by_caseids_prerisk/ANS2E_IOH/"

folder_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("folder_")]
raw_files = []
for folder_dir in folder_dirs:
    files_in_folder = [f for f in os.listdir(folder_dir) if f.endswith("_timeseries.csv")]
    raw_files.extend([os.path.join(folder_dir, f) for f in files_in_folder])

cases_csv_path = "/home/share/datasets/ioh/VitalDB/vitaldb_oridataset/webapi_dataset/cases.csv"
output_json = "/home/share/datasets/ioh/VitalDB/processed/timeseries_by_caseids_prerisk/ANS2E_IOH/vitals_ioh_timeline.json"

df_cases = pd.read_csv(cases_csv_path)
df_cases["caseid"] = df_cases["caseid"].astype(int)

rows = []
for file in raw_files:
    file_name = os.path.basename(file)
    if not file_name.endswith("_timeseries.csv"):
        continue
    try:
        caseid = int(file_name.split("_")[0])
        meta = df_cases.loc[df_cases["caseid"] == caseid, ["anestart", "aneend"]]
        if meta.empty:
            continue

        df = pd.read_csv(file)
        if df.empty or df.shape[1] < 2:
            continue

        time_values = df.iloc[:, 0].tolist()

        row = {
            "caseid": caseid,
            "anestart": meta["anestart"].values[0],
            "aneend": meta["aneend"].values[0],
            "chart_times": time_values,
        }
        for col in df.columns[1:]:
            row[col] = df[col].tolist()
        rows.append(row)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

rows_sorted = sorted(rows, key=lambda x: x["caseid"])

def prepare_for_json(obj):
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            val = obj.item()
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
            return val
        def _clean_scalar(x):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return x
        return [_clean_scalar(x) for x in obj.tolist()]
    elif isinstance(obj, (list, tuple)):
        return [prepare_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: prepare_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (np.float64, np.float32)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj

rows_prepared = prepare_for_json(rows_sorted)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(rows_prepared, f, ensure_ascii=False, indent=4, allow_nan=False)

print(f"Saved to {output_json}")