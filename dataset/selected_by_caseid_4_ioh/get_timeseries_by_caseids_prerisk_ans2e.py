import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from utils import get_config
from utils import round_data_to_two_decimals, parse_arguments
from process import check_and_load_dataset_csv, load_timeseries_dict

import time

def setup_paths(config):
    tname_selection = config['tname_selection']
    trks_csv_path = config['paths']['trks_csv_path']
    cases_csv_path = config['paths']['cases_csv_path']
    labs_csv_path = config['paths']['labs_csv_path']
    tids_folder_path = config['paths']['tids_folder_path']
    ioh_dataset_csv_path = config['paths']['ioh_dataset_csv_path']


    # 确保要保存的dataset和日志文件夹存在
    try:
        dataset_folder_path = os.path.dirname(ioh_dataset_csv_path)
        os.makedirs(dataset_folder_path, exist_ok=True)
        # os.makedirs(ioh_dataset_log_folder_path, exist_ok=True)
    except OSError as e:
        print(f"创建目录时出错: {e}")

    return tname_selection, trks_csv_path, cases_csv_path, labs_csv_path, tids_folder_path, ioh_dataset_csv_path


def get_total_cases(trks_csv_path, tname_selection):

    df_trks = pd.read_csv(trks_csv_path)

    # 根据配置文件中的tname_selection提取caseids
    selected_tnames = [tname for tname, selected in tname_selection.items() if selected]
    caseids = list(
        set.intersection(*[
            set(df_trks[df_trks['tname'] == tname]['caseid'])
            for tname in selected_tnames
        ])
    )

    print('Combined {}, Total {} cases found'.format((selected_tnames), len(caseids)))

    return caseids

def process_caseids(dataset_name, caseids, tname_selection, config, cases_csv_path, timeseries_dataset_csv_path):

    selected_tnames = [tname for tname, selected in tname_selection.items() if selected]

    STIME = config['parameters']['STIME']   # 采样间隔（秒），通常为2秒
    

    df_cases = pd.read_csv(cases_csv_path)
    # df_trks = pd.read_csv(trks_csv_path)
    # print("df_trks:", df_trks.shape)

    total_caseids = len(caseids)
    valid_cases = 0
    # 定义每个文件夹保存的 caseid 数量, 用于分工人工数据分析
    CASES_PER_FOLDER = 500
    for index, caseid in enumerate(caseids, start=1):
        print(f'loading {caseid} ({index}/{total_caseids})', end='...\n', flush=True)

        timeseries_dict = load_timeseries_dict(caseid, selected_tnames, STIME)
        
        # 对于任意的整条记录，如果缺失率如果大于20%，则不用该caseid
        # 检查每个时间序列的缺失率
        high_missing_rate_series = [
            name for name, timeseries in timeseries_dict.items()
            if np.isnan(timeseries).mean() > 0.2
        ]
        
        if high_missing_rate_series:
            print(f"Case {caseid} skipped due to high missing rate in: {', '.join(high_missing_rate_series)}")
            continue
        
        # 术前风险评估定义的术前是麻醉开始时间
        # Get the current case's age, gender, height, and weight
        aswh_pd = df_cases.loc[df_cases['caseid'] == caseid, ['age', 'sex', 'bmi', 'opstart', 'opend', 'anestart', 'aneend']]
        aswh = [aswh_pd['age'].values[0], 0 if aswh_pd['sex'].values[0].lower() == "m" else 1, aswh_pd['bmi'].values[0]]

        
        # # TODO 直接把该caseid的全部序列数据进行保存
        min_len = min(len(maps) for maps in timeseries_dict.values())
        
        # 创建时间列
        time = np.arange(0, min_len) * STIME + STIME  
        df = pd.DataFrame({'time': time})
        
        # 将每个时间序列添加到 DataFrame 中
        for name, timeseries in timeseries_dict.items():
            df[name] = np.round(timeseries[:min_len], 2) 
            
        # 根据手术开始（opstart）和结束（opend）时间截取数据
        opstart_val = aswh_pd['opstart'].values[0]
        opend_val = aswh_pd['opend'].values[0]
        anestart_val = aswh_pd['anestart'].values[0]
        aneend_val = aswh_pd['aneend'].values[0]

        # 基于麻醉时长纳排：若麻醉结束-麻醉开始 < 1 小时，则跳过
        ane_duration = aneend_val - anestart_val
        if pd.isna(ane_duration) or ane_duration < 3600:
            print(f"Case {caseid} 因麻醉时长不足1小时被跳过（{ane_duration:.0f}秒）")
            continue

        # 按麻醉开始/结束截取数据
        # df = df[(df['time'] >= opstart_val) & (df['time'] <= opend_val)]
        df = df[(df['time'] >= anestart_val) & (df['time'] <= aneend_val)]
        
        # 若截取后没有数据也跳过
        if len(df) == 0:
            print(f"Case {caseid} 在麻醉时间窗内无数据，跳过")
            continue
        
        valid_cases += 1
        
        # 动态生成文件夹路径
        folder_index = (valid_cases - 1) // CASES_PER_FOLDER + 1  # 计算属于第几个文件夹
        temp_folder = f"{timeseries_dataset_csv_path}/folder_{folder_index}"  # 文件夹名称
        os.makedirs(temp_folder, exist_ok=True)  # 确保文件夹存在

        # 构造输出文件路径
        output_file = os.path.join(temp_folder, f"{caseid}_timeseries.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {caseid} to {output_file}")
        
        # if valid_cases == 5:
        #     exit()
    print(f"本次抽取共计 {valid_cases} 个有效 caseid")
        
if __name__ == "__main__":
    start_time = time.time()  # 开始时间戳
    # 使用示例：
    # eg：nohup python -u get_timeseries_by_caseids_prerisk_ans2e.py --dataset_name timeseries_by_caseids_prerisk_ans2e_ioh > ../output/logs/output_get_timeseries_by_caseids_prerisk_ans2e_ioh.log 2>&1 &
    args = parse_arguments()
    dataset_name = args.dataset_name
    config = get_config(dataset_name)
    tname_selection, trks_csv_path, cases_csv_path, labs_csv_path, tids_folder_path, timeseries_dataset_csv_path = setup_paths(config)
    caseids = get_total_cases(trks_csv_path, tname_selection)
    selected_tnames = [tname for tname, selected in tname_selection.items() if selected]
    
    # exit()

    # # 以caseid=3为例查看csv数据长什么样
    # tids = load_case_tids(trks_csv_path, tids_folder_path, caseid=3, track_names=selected_tnames)
    # down_load_trks(tids_folder_path, tids, selected_tnames)
    # Check if the CSV file already exists, and load previously saved caseid data if available
    
    # # 从caseids去掉之前已经处理过的caseid
    # max_processed_caseids = check_and_load_dataset_csv(ioh_dataset_csv_path)
    # caseids = [caseid for caseid in caseids if caseid > max_processed_caseids]

    process_caseids(dataset_name=dataset_name, 
                      caseids=caseids, 
                      tname_selection=tname_selection, 
                      config=config, 
                      cases_csv_path=cases_csv_path, 
                      timeseries_dataset_csv_path=timeseries_dataset_csv_path)
    print('Done!')
    
    end_time = time.time()  # 结束时间戳
    elapsed_time = (end_time - start_time) / 60  # 计算耗时并转换为分钟
    print(f"程序执行时间: {elapsed_time:.2f} 分钟")
