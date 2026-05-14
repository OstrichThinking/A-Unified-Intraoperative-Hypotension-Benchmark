import os
from pathlib import Path
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
import data_prep.VitalDB.vitaldb_local as vitaldb


dftrks = None
dfci = None

def load_static_info(cases_csv_path, caseid, basic_info_fields):
    """Get basic information for a specific case ID.

    Parameters:
        cases_csv_path (str): Path to the CSV file containing case data.
        caseid (int): The ID of the case to retrieve information for.
        basic_info_fields (list): A list of fields to extract for basic information.

    Returns:
        case_info (list): A list containing the basic information of the case.
    """
    global dfci
    if dfci is None:
        dfci = pd.read_csv(cases_csv_path)

    if not caseid:
        return None

    case_info = dfci.loc[dfci['caseid'] == caseid, basic_info_fields]

    return case_info.values.tolist()[0]

def check_trk(tids_folder_path, tid):
    """
    Check if the track file exists and is not empty.

    Parameters:
        tids_folder_path (str): The path to the folder containing track files.
        tid (str): The track ID to check.

    Returns:
        str or None: Returns the track ID if the file exists and is not empty, otherwise returns None.
    """
    tid_path = f"{tids_folder_path}/{tid}.csv"

    try:
        # 如果csv文件不存在，则报错
        if not os.path.exists(tid_path):
            raise FileNotFoundError(f"{tid_path} does not exist")
        
        # 读取CSV文件
        dtvals = pd.read_csv(tid_path, na_values='-nan(ind)', dtype=np.float32).values
        
        # 如果dtvals为空，则抛出警告
        if len(dtvals) == 0:
            print(f"Warning: {tid_path} is empty")
            return None
    except FileNotFoundError as e:
        # 捕获文件未找到错误并打印警告
        print(f"Warning: {e}")
        return None
    except pd.errors.EmptyDataError:
        # 捕获空数据错误并打印警告
        print(f"Warning: {tid_path} is empty")
        return None
    except Exception as e:
        # 捕获其他异常并打印警告
        print(f"An error occurred: {e}")
        return None

    return tid

def load_case_tids(trks_csv_path, tids_folder_path, caseid, track_names):
    """Load case data with the given track names in a 2D numpy array. 

    Parameters:
        caseid (int): caseID from 1 to 6388
        track_names (list or string):  a list of track names or a string with track names separated by comma

    Returns:
        tids (list): tids of the tracks
    """
    global dftrks

    if not caseid:
        return None

    if dftrks is None:
        dftrks = pd.read_csv(trks_csv_path)

    if isinstance(track_names, str):
        if track_names.find(','):
            track_names = track_names.split(',')
        else:
            track_names = [track_names]

    tids = []
    for dtname in track_names:
        tid_values = dftrks.loc[(dftrks['caseid'] == caseid) & (dftrks['tname'].str.endswith(dtname)), 'tid'].values
        if len(tid_values):
            tid = check_trk(tids_folder_path, tid_values[0])
            tids.append(tid)
        else:
            tids.append(None)
    
    return tids

def down_load_trks(tids_folder_path, tids, selected_tnames):
    output_folder_path = Path(__file__).parent.parent / "output/samples/Invasive_group"    # 指定输出目录
    os.makedirs(output_folder_path, exist_ok=True)  # 确保输出目录存在

    for id, tid in enumerate(tids):
        if tid is None:
            continue
        trk_path = f"{tids_folder_path}/{tid}.csv"
        if not os.path.exists(trk_path):
            print(f"Warning: {trk_path} does not exist")
            return None
        
        # 读取 CSV 文件
        data = pd.read_csv(trk_path, na_values='-nan(ind)', dtype=np.float32)
        # data = pd.read_csv(trk_path, na_values='-nan(ind)', dtype=np.float32).values
        
        # 替换 selected_tnames[id] 中的斜杠为下划线
        re_tname = selected_tnames[id].replace('/', '_')
        
        # 保存到新的目录
        output_path = f"{output_folder_path}/{id}_{re_tname}_{tid}.csv"
        pd.DataFrame(data).to_csv(output_path, index=False)


def check_and_load_dataset_csv(file_path):
    """
    检查CSV文件是否存在且不为空，并返回已处理的caseid列表和最大caseid。
    
    参数:
    - file_path: str, CSV文件的路径
    
    返回:
    - processed_caseids: list, 已处理的caseid列表
    - max_processed_caseids: int, 已处理的最大caseid
    """
    max_processed_caseids = 0
    processed_caseids = []

    # 检查文件是否存在
    if os.path.exists(file_path):
        # 检查文件是否为空
        if os.path.getsize(file_path) == 0:
            # 如果文件为空，删除文件
            os.remove(file_path)
            print(f"文件 {file_path} 是空的，已被删除。")
        else:
            try:
                # 如果文件不为空，尝试读取现有数据
                existing_data = pd.read_csv(file_path)
                if existing_data.empty:
                    raise pd.errors.EmptyDataError("文件内容为空")
                processed_caseids = existing_data['caseid'].unique().tolist()
                max_processed_caseids = max(processed_caseids)
                print(f"文件 {file_path} 存在且不为空，已读取数据。")
            except pd.errors.EmptyDataError:
                print(f"文件 {file_path} 存在但内容为空，已被删除。")
                os.remove(file_path)
    else:
        print(f"文件 {file_path} 不存在。")

    return max_processed_caseids


def load_timeseries_dict(caseid, selected_tnames, STIME):
    """
    加载指定的时间序列数据并返回一个字典。

    参数：
    - dataset_name: 数据集名称
    - caseid: 病例ID
    - SurgeryStatus_time: 手术状态时间
    - selected_tnames: 选择的时间序列名称列表
    - STIME: 抽样周期

    返回：
    - timeseries_dict: 包含所有时间序列数据的字典
    """
    timeseries_dict = {
        tname: vitaldb.load_case(caseid, [tname], STIME).flatten()
        for tname in selected_tnames
    }
    return timeseries_dict