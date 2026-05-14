import json
import os
import random
import textwrap
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

IS_DEBUG = False
JSONL_LINES_TEST = 2


def read_data(file_path):
    """读取 JSONL 并按病例划分 train/test/val (7:2:1)。"""
    case_list = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            line_count = 0
            for line in file:
                if line_count >= JSONL_LINES_TEST and IS_DEBUG:
                    break
                try:
                    case = json.loads(line.strip())
                    case_list.append(case)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {e}")
                    continue
                line_count += 1
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading JSONL file: {e}") from e

    random.shuffle(case_list)

    n_caseids = len(case_list)
    train_cut = int(n_caseids * 0.7)
    test_cut = train_cut + int(n_caseids * 0.2)

    case_subset_train = case_list[:train_cut]
    case_subset_test = case_list[train_cut:test_cut]
    case_subset_val = case_list[test_cut:]
    return case_subset_train, case_subset_test, case_subset_val


def _to_float_array(ts):
    """将列表转换为 float 数组，None/非法值转为 NaN。"""
    if ts is None:
        return np.array([], dtype=float)
    values = []
    for v in ts:
        if v is None:
            values.append(np.nan)
        else:
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                values.append(np.nan)
    return np.array(values, dtype=float)


def _resample_series(ts_array, case_stime, exp_stime):
    """当原始采样间隔更密时，做简单整倍数降采样。"""
    if case_stime == exp_stime:
        return ts_array
    if case_stime <= 0 or exp_stime <= 0:
        return None
    if exp_stime % case_stime != 0:
        return None
    ratio = exp_stime // case_stime
    if ratio <= 0:
        return None
    return ts_array[::ratio]


def validate_and_align_case_time(case, dynamic_features, exp_stime, min_seconds=3600):
    """
    1. 仅保留动态波形特征，并统一为 float 数组
    2. 按 ART_MBP 去掉前后全缺失区间
    3. 按目标采样间隔降采样并长度对齐
    """
    case_stime = int(case.get("stime", exp_stime))
    if case_stime <= 0:
        return None

    converted = {}
    lengths = []
    for feature in dynamic_features:
        arr = _to_float_array(case.get(feature, []))
        if arr.size == 0:
            return None
        converted[feature] = arr
        lengths.append(arr.size)
    if len(set(lengths)) != 1:
        return None

    mbps = converted["ART_MBP"]
    valid_mask = ~np.isnan(mbps)
    if not valid_mask.any():
        return None

    first_valid = int(np.argmax(valid_mask))
    last_valid = int(len(valid_mask) - 1 - np.argmax(valid_mask[::-1]))
    if last_valid < first_valid:
        return None

    aligned = {}
    for feature in dynamic_features:
        trimmed = converted[feature][first_valid : last_valid + 1]
        downsampled = _resample_series(trimmed, case_stime, exp_stime)
        if downsampled is None or downsampled.size == 0:
            return None
        aligned[feature] = downsampled

    aligned_lengths = [len(aligned[k]) for k in dynamic_features]
    if len(set(aligned_lengths)) != 1:
        return None

    min_len = int(min_seconds // exp_stime)
    if aligned_lengths[0] < min_len:
        return None
    return aligned


def _fill_nan_1d(arr):
    """对 1D 序列做前后向填充；全 NaN 时返回 None。"""
    if arr.size == 0:
        return None
    series = pd.Series(arr, dtype=float).ffill().bfill()
    if series.isna().all():
        return None
    return series.values


def risk_strength(map_values):
    """
    时刻级风险强度 r(t):
    MAP > 75 -> 0
    MAP < 65 -> 1
    65 <= MAP <= 75 -> (75 - MAP) / 10
    """
    r = np.zeros_like(map_values, dtype=float)
    low_mask = map_values < 65
    high_mask = map_values > 75
    gray_mask = (~low_mask) & (~high_mask)
    r[low_mask] = 1.0
    r[gray_mask] = (75.0 - map_values[gray_mask]) / 10.0
    return np.clip(r, 0.0, 1.0)


def calculate_R60(window):
    """对窗口内 MAP 计算平均风险分数 (0~1)。"""
    filled = _fill_nan_1d(window)
    if filled is None:
        return None
    return float(np.mean(risk_strength(filled)))


def _has_continuous(window, threshold, op, min_samples):
    """检查是否存在满足条件的连续区间。"""
    if op == "lt":
        cond = window < threshold
    elif op == "gt":
        cond = window > threshold
    else:
        raise ValueError(f"Unsupported op: {op}")

    run = 0
    for flag in cond:
        if flag:
            run += 1
            if run >= min_samples:
                return True
        else:
            run = 0
    return False


def check_hypotension_continuous(window, exp_stime):
    """未来窗口内是否出现持续 1 分钟低血压 (MAP < 65)。"""
    filled = _fill_nan_1d(window)
    if filled is None:
        return False
    min_samples = int(np.ceil(60 / exp_stime))
    return _has_continuous(filled, 65, "lt", min_samples)


def check_normal_continuous(window, exp_stime):
    """未来窗口内是否出现持续 1 分钟非低血压 (MAP > 75)。"""
    filled = _fill_nan_1d(window)
    if filled is None:
        return False
    min_samples = int(np.ceil(60 / exp_stime))
    return _has_continuous(filled, 75, "gt", min_samples)


def calculate_trend_slope(window, exp_stime, smooth_seconds=6):
    """
    对 MAP 做短窗中值滤波后计算趋势斜率，单位 mmHg/min。
    """
    filled = _fill_nan_1d(window)
    if filled is None or len(filled) < 2:
        return None

    smooth_points = max(2, int(np.ceil(smooth_seconds / exp_stime)))
    denoised = (
        pd.Series(filled).rolling(window=smooth_points, min_periods=1, center=True).median().values
    )
    x_minutes = np.arange(len(denoised), dtype=float) * (exp_stime / 60.0)
    slope = np.polyfit(x_minutes, denoised, 1)[0]
    return float(slope)


def assign_label(window, exp_stime, slope_threshold):
    """
    四分类优先级:
    3: 持续 1 分钟低血压
    0: 持续 1 分钟非低血压
    2: 灰区高风险 (R_score > 0.5 或 slope 下降)
    1: 灰区低风险 (剩余情况)
    """
    r_score = calculate_R60(window)
    slope = calculate_trend_slope(window, exp_stime=exp_stime)
    if r_score is None or slope is None:
        return None

    is_hk = check_hypotension_continuous(window, exp_stime=exp_stime)
    if is_hk:
        return 3

    is_nk = check_normal_continuous(window, exp_stime=exp_stime)
    if is_nk:
        return 0

    # 文档语义为“且”逻辑：R 高且趋势下降都归为灰区高风险
    slope_eps = 1e-6
    slope_threshold = max(float(slope_threshold), slope_eps)
    if (r_score > 0.5) and (slope < -slope_threshold):
        return 2
    return 1


def get_label_details(window, exp_stime, slope_threshold):
    """
    返回标签以及生成标签所需的中间指标，便于逐样本打印。
    """
    r_score = calculate_R60(window)
    slope = calculate_trend_slope(window, exp_stime=exp_stime)
    if r_score is None or slope is None:
        return None

    is_hk = check_hypotension_continuous(window, exp_stime=exp_stime)
    is_nk = check_normal_continuous(window, exp_stime=exp_stime)
    label = assign_label(window, exp_stime=exp_stime, slope_threshold=slope_threshold)
    if label is None:
        return None

    return {
        "label": int(label),
        "R_score": float(r_score),
        "slope": float(slope),
        "is_Hk": bool(is_hk),
        "is_Nk": bool(is_nk),
    }


def create_segment_list(case, seq_len, start_idx, feature):
    x_start = start_idx - seq_len
    x_end = start_idx
    return np.array(case[feature][x_start:x_end], dtype=float)


def convert_jsonl_to_sample_list(
    case_subset,
    flag,
    seq_len,
    pred_len,
    slide_len,
    exp_stime,
    dynamic_features,
    slope_threshold=0.0,
    print_label_samples=False,
    gap_len=0,
):
    sample_list = defaultdict(list)
    labels = []

    for case in tqdm(case_subset[:], desc=f"{flag} cases"):
        case_id = case.get("id", "unknown")
        aligned_case = validate_and_align_case_time(
            case,
            dynamic_features=dynamic_features,
            exp_stime=exp_stime,
        )
        if aligned_case is None:
            continue

        case_len = len(aligned_case["ART_MBP"])
        start_min = seq_len + gap_len
        start_max = case_len - pred_len
        if start_max < start_min:
            continue

        for pred_start_idx in range(start_min, start_max + 1, slide_len):
            obs_end_exclusive = pred_start_idx - gap_len
            pred_window = aligned_case["ART_MBP"][pred_start_idx : pred_start_idx + pred_len]
            label_details = get_label_details(
                pred_window,
                exp_stime=exp_stime,
                slope_threshold=slope_threshold,
            )
            if label_details is None:
                continue
            label = label_details["label"]

            is_valid_len = True
            segments = {}
            for feature in dynamic_features:
                seg = create_segment_list(
                    aligned_case,
                    seq_len=seq_len,
                    start_idx=obs_end_exclusive,
                    feature=feature,
                )
                if len(seg) != seq_len:
                    is_valid_len = False
                    break
                segments[feature] = seg
            if not is_valid_len:
                continue

            for feature in dynamic_features:
                sample_list[feature].append(segments[feature])
            labels.append(label)

            if print_label_samples:
                obs_start = obs_end_exclusive - seq_len
                obs_end = obs_end_exclusive - 1
                gap_start = obs_end + 1
                gap_end = pred_start_idx - 1
                pred_start = pred_start_idx
                pred_end = pred_start_idx + pred_len - 1
                sample_info = {
                    "subset": flag,
                    "case_id": case_id,
                    "sample_idx": len(labels) - 1,
                    "obs_range_idx": [obs_start, obs_end],
                    "gap_range_idx": [gap_start, gap_end],
                    "pred_range_idx": [pred_start, pred_end],
                    "label": label_details["label"],
                    "R_score": round(label_details["R_score"], 6),
                    "slope": round(label_details["slope"], 6),
                    "is_Hk": label_details["is_Hk"],
                    "is_Nk": label_details["is_Nk"],
                    "obs_ART_MBP": np.round(aligned_case["ART_MBP"][obs_start:obs_end_exclusive], 2).tolist(),
                    "pred_ART_MBP": np.round(pred_window, 2).tolist(),
                }
                # print("[LABEL_SAMPLE]", json.dumps(sample_info, ensure_ascii=False))

    n_before = len(labels)
    print(f"{flag} 样本检查前: {n_before}")
    sample_list, labels = check_sample_valid(sample_list, labels, dynamic_features)
    print(f"{flag} 样本检查后有效: {len(labels)}")
    if labels:
        label_counts = {k: labels.count(k) for k in [0, 1, 2, 3]}
        print(f"{flag} 标签分布: {label_counts}")
    return sample_list, labels


def convert_sample_list_to_ts(sample_list, labels, path, dynamic_features, seq_len):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    meta_data = textwrap.dedent(
        f"""\
    # This file is converted from JSONL to TS format from VitalDB dataset, for IOH 4-class classification task.
    @problemName VitalDB-IOH-4Class
    @timeStamps false
    @missing false
    @univariate false
    @dimensions {len(dynamic_features)}
    @equalLength true
    @seriesLength {seq_len}
    @classLabel true 0 1 2 3
    @data
    """
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(meta_data)
        if not labels:
            return
        for i in range(len(labels)):
            lines = []
            for feature in dynamic_features:
                line = ",".join(str(round(value, 2)) for value in sample_list[feature][i])
                lines.append(line)
            full_line = ":".join(lines) + f":{int(labels[i])}\n"
            f.write(full_line)


def check_sample_valid(sample_list, labels, dynamic_features):
    """
    样本质量检查 + 动态特征缺失值填充。
    """
    if not sample_list:
        return defaultdict(list), []

    lengths = [len(sample_list[key]) for key in sample_list]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"sample_list lengths are inconsistent: {dict((k, len(sample_list[k])) for k in sample_list)}"
        )

    total_samples = lengths[0]
    if total_samples == 0:
        return defaultdict(list), []

    if len(labels) != total_samples:
        raise ValueError(f"labels size mismatch: labels={len(labels)}, samples={total_samples}")

    data_dict = {}
    for key in sample_list:
        data_dict[key] = np.array(sample_list[key], dtype=float)

    valid_mask = np.ones(total_samples, dtype=bool)

    for feature in dynamic_features:
        if feature in data_dict:
            has_neg = (data_dict[feature] < 0).any(axis=1)
            valid_mask &= ~has_neg

            nan_ratio = np.isnan(data_dict[feature]).mean(axis=1)
            valid_mask &= nan_ratio <= 0.2

    if "ART_MBP" in data_dict:
        nan_ratio = np.isnan(data_dict["ART_MBP"]).mean(axis=1)
        valid_mask &= nan_ratio <= 0.1

        diffs = np.abs(np.diff(data_dict["ART_MBP"], axis=1))
        has_large_diff = (diffs > 50).any(axis=1)
        valid_mask &= ~has_large_diff

    valid_sample_list = defaultdict(list)
    if not valid_mask.any():
        return valid_sample_list, []

    valid_labels = np.array(labels, dtype=int)[valid_mask].tolist()
    for key, data in data_dict.items():
        valid_data = data[valid_mask]
        if key in dynamic_features:
            df = pd.DataFrame(valid_data)
            filled_data = df.ffill(axis=1).bfill(axis=1).values
            valid_sample_list[key] = list(filled_data)
        else:
            valid_sample_list[key] = list(valid_data)
    return valid_sample_list, valid_labels


if __name__ == "__main__":
    # nohup python VitalDB_1810/cover_jsonl_to_ts_4_classificaiton.py > cover_jsonl_to_ts_4_classificaiton.log 2>&1 &
    fix_seed = 42
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    jsonl_path = "data/VitalDB_1810/ioh_dataset_vitaldb.jsonl"
    ts_path = "data/VitalDB_1810/cls_4/"

    dynamic_features = ["ETCO2", "VENT_MAWP", "HR", "ART_DBP", "ART_SBP", "ART_MBP"]

    exp_stime = 2  # 2秒一个点
    observe_minutes = 15
    slide_minutes = 2
    gap_minutes = 2
    predict_minutes = 5
    slope_threshold = 0.0
    print_label_samples = True

    seq_len = int(observe_minutes * 60 / exp_stime)
    slide_len = int(slide_minutes * 60 / exp_stime)
    gap_len = int(gap_minutes * 60 / exp_stime)
    pred_len = int(predict_minutes * 60 / exp_stime)

    case_subset_train, case_subset_test, case_subset_val = read_data(jsonl_path)

    for case_subset, flag in zip(
        (case_subset_train, case_subset_test, case_subset_val),
        ("train", "test", "val"),
    ):
        sample_list, labels = convert_jsonl_to_sample_list(
            case_subset=case_subset,
            flag=flag,
            seq_len=seq_len,
            pred_len=pred_len,
            slide_len=slide_len,
            gap_len=gap_len,
            exp_stime=exp_stime,
            dynamic_features=dynamic_features,
            slope_threshold=slope_threshold,
            print_label_samples=print_label_samples,
        )
        out_path = os.path.join(ts_path, f"vitaldb_{flag}.ts")
        convert_sample_list_to_ts(
            sample_list=sample_list,
            labels=labels,
            path=out_path,
            dynamic_features=dynamic_features,
            seq_len=seq_len,
        )
        print(f"{out_path} conversion completed.")
