# -*- coding: utf-8 -*-
"""
出租车GPS数据清洗模块
======================
对应阶段：01 项目导入与清洗启动 / 02 清洗完成与OD启动
依据文档：《数据简单处理.docx》需求1-11、《实训V_项目介绍.docx》数据清洗模块要求

字段说明（原始数据无表头）：
    id     车辆编号
    time   GPS采集时间（仅有时分秒，需要拼接日期）
    long   经度
    lati   纬度
    status 载客状态（1=载客，0=空载）
    speed  GPS车速

输出：
    清洗后的数据文件（统一命名、统一列名），供后续 OD 提取、缓存构建直接读取。

使用方式：
    1. 修改下方“配置区”的路径（不要写死个人电脑的绝对路径，建议放到项目根目录下的相对路径）
    2. 直接运行：python data_cleaning.py
    3. 如果数据量过大（千万级），把 USE_CHUNK 设为 True，按 CHUNK_SIZE 分块读取
"""

import os
import logging
import pandas as pd
import numpy as np

# ========================= 配置区（唯一需要手动改的地方） =========================
RAW_DATA_PATH = "../data/raw/TaxiData.csv"          # 原始数据路径（相对项目根目录）
CLEAN_DATA_PATH = "../data/processed/taxi_clean.csv"  # 清洗后数据输出路径
OD_DATA_PATH = "../data/processed/taxi_od.csv"        # OD（上下车点）提取结果输出路径
LOG_PATH = "../docs/cleaning_log.txt"            # 清洗日志输出路径

COLUMNS = ['id', 'time', 'long', 'lati', 'status', 'speed']  # 统一列名，全项目必须保持一致
DATE_PREFIX = "2013-10-22"   # 原始time只有时分秒，需要拼接的日期；按实际数据日期修改
ABNORMAL_SECONDS_THRESHOLD = 60  # 异常状态切换的时间阈值（需求8-10）

USE_CHUNK = False             # 数据量是千万级时改为 True
CHUNK_SIZE = 5_000_000        # 分块大小，按机器内存调整
# ===============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# --------------------------- 1. 数据读取 ---------------------------
def load_data(path: str, use_chunk: bool = False, chunk_size: int = 5_000_000) -> pd.DataFrame:
    """
    读取原始GPS数据。原数据没有header，因此header=None并手动指定列名。
    数据量过大时使用chunksize分块读取，最后拼接为一个DataFrame。
    注意：分块只是为了降低单次内存占用，去重/排序/异常检测仍需要在全量数据上进行，
    所以这里分块读取后会拼接为完整df，而不是逐块独立清洗（避免同一车辆数据跨分块导致误判）。
    """
    if not use_chunk:
        df = pd.read_csv(path, header=None, names=COLUMNS)
        log.info(f"一次性读取数据完成，shape={df.shape}")
        return df

    chunks = []
    reader = pd.read_csv(path, header=None, names=COLUMNS, chunksize=chunk_size)
    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        log.info(f"读取第{i+1}个分块，shape={chunk.shape}")
    df = pd.concat(chunks, ignore_index=True)
    log.info(f"全部分块拼接完成，总shape={df.shape}")
    return df


# --------------------------- 2. 基础类型与排序（需求1-2） ---------------------------
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    需求1：按 id, time 升序排序并重置索引
    需求2：time 转换为时间戳类型（原数据time只有时分秒，需先拼接日期）
    """
    before = df.shape[0]

    # 原始time可能只有"HH:MM:SS"，拼接日期后转换，避免使用当前日期造成所有数据日期不一致
    if df['time'].astype(str).str.match(r'^\d{1,2}:\d{2}:\d{2}$').all():
        df['time'] = DATE_PREFIX + ' ' + df['time'].astype(str)

    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # 关键字段缺失（id/time/long/lati/status缺失）直接判定为脏数据
    na_mask = df[['id', 'time', 'long', 'lati', 'status']].isna().any(axis=1)
    if na_mask.any():
        log.info(f"发现关键字段缺失记录 {na_mask.sum()} 条，已剔除")
        df = df.loc[~na_mask]

    df = df.sort_values(by=['id', 'time']).reset_index(drop=True)

    log.info(f"基础清洗（类型转换+排序）完成：{before} -> {df.shape[0]}")
    return df


# --------------------------- 3. 非法值/越界值清洗（项目介绍中的"丢弃异常数据"） ---------------------------
def drop_invalid_records(
    df: pd.DataFrame,
    lng_range=(113.7, 114.7),   # 按实际城市范围修改，这里以深圳为示例区间
    lat_range=(22.4, 22.9),
    max_speed=120
) -> pd.DataFrame:
    """
    剔除明显不合法的记录：
        - 坐标超出城市范围
        - 车速为负或超过城市道路常规上限（如120km/h）
        - status 不是 0/1
    """
    before = df.shape[0]

    cond_lng = df['long'].between(*lng_range)
    cond_lat = df['lati'].between(*lat_range)
    cond_speed = df['speed'].between(0, max_speed)
    cond_status = df['status'].isin([0, 1])

    valid_mask = cond_lng & cond_lat & cond_speed & cond_status
    dropped = before - valid_mask.sum()
    df = df.loc[valid_mask].reset_index(drop=True)

    log.info(f"非法值/越界值清洗：剔除 {dropped} 条，{before} -> {df.shape[0]}")
    return df


def drop_always_same_status_vehicles(df: pd.DataFrame) -> pd.DataFrame:
    """
    剔除"全天未载客"或"全天载客中"的异常车辆（项目介绍中的丢弃异常数据要求）。
    这类车辆的数据虽然没有缺失或越界，但status全天没有任何变化，通常说明设备故障。
    """
    before_n = df['id'].nunique()
    status_var = df.groupby('id')['status'].nunique()
    abnormal_ids = status_var[status_var == 1].index
    df = df.loc[~df['id'].isin(abnormal_ids)].reset_index(drop=True)
    log.info(f"剔除status全天不变的车辆：{len(abnormal_ids)} 辆（共 {before_n} 辆），剩余 {df['id'].nunique()} 辆")
    return df


# --------------------------- 4. 重复值清洗（需求3-7） ---------------------------
def _dup_check(group: pd.DataFrame):
    """
    需求7的判断逻辑：根据重复数量(stat_cnt)和status求和(stat_sum)判断保留哪一行索引。
    少数服从多数：status多数为0就保留一个0；多数为1就保留一个1。
    """
    cnt, s = group['stat_cnt'].max(), group['stat_sum'].max()
    if s == 0:
        return group['index'].values[0]
    if s == cnt:
        return group['index'].values[0]
    if s < cnt / 2:
        return group.loc[group['status'] == 0, 'index'].values[0]
    else:
        return group.loc[group['status'] == 1, 'index'].values[0]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    需求3：找出 id+time 重复的全部记录
    需求4：统计每组重复数量，检查是否存在大于2的情况
    需求5：筛选出重复数量大于2的数据单独看一下分布
    需求6：对每组 status 求 count 和 sum，合并回重复数据
    需求7：按规则筛选出应保留的索引，从原数据中剔除其余的重复行
    """
    before = df.shape[0]
    df_dup = df[df.duplicated(subset=['id', 'time'], keep=False)].reset_index()

    if df_dup.empty:
        log.info("未发现 id+time 重复数据")
        return df

    log.info(f"发现 id+time 重复数据 {df_dup.shape[0]} 条")

    dup_cnt = df_dup.groupby(['id', 'time'])['status'].count()
    all_two = (dup_cnt == 2).all()
    log.info(f"重复数量是否全部为2：{all_two}")
    if not all_two:
        more_than_two = dup_cnt[dup_cnt > 2]
        log.info(f"重复数量大于2的组合数：{len(more_than_two)}，最大重复数量：{dup_cnt.max()}")

    dup_grp = (
        df_dup.groupby(['id', 'time'])
              .agg(stat_cnt=('status', 'count'), stat_sum=('status', 'sum'))
              .reset_index()
    )
    dup_mrg = pd.merge(df_dup, dup_grp, on=['id', 'time'], how='left')

    kp_index = dup_mrg.groupby(['id', 'time']).apply(_dup_check)
    drp_index = dup_mrg.loc[~dup_mrg['index'].isin(kp_index.values), 'index']

    df = df.loc[~df.index.isin(drp_index.values)].reset_index(drop=True)
    log.info(f"去重完成：{before} -> {df.shape[0]}（剔除 {before - df.shape[0]} 条）")
    return df


# --------------------------- 5. 异常状态切换清洗（需求8-10） ---------------------------
def remove_abnormal_status(df: pd.DataFrame, seconds_threshold: int = ABNORMAL_SECONDS_THRESHOLD) -> pd.DataFrame:
    """
    需求8：构造前后状态/前后车辆id/前后时间（shift平移）
    需求9：筛选出短时间内状态突变（0-1-0 或 1-0-1）的异常记录
    需求10：从原数据中剔除这些异常记录
    """
    before = df.shape[0]

    df['status_up'] = df['status'].shift(1)
    df['status_down'] = df['status'].shift(-1)
    df['id_up'] = df['id'].shift(1)
    df['id_down'] = df['id'].shift(-1)
    df['time_up'] = df['time'].shift(1)
    df['time_down'] = df['time'].shift(-1)

    cond_1 = df['status'] != df['status_down']
    cond_2 = df['status'] != df['status_up']
    cond_3 = df['id'] == df['id_up']
    cond_4 = df['id'] == df['id_down']
    cond_5 = (df['time_down'] - df['time_up']).dt.seconds < seconds_threshold

    df_abn = df[cond_1 & cond_2 & cond_3 & cond_4 & cond_5].reset_index()
    log.info(f"短时间状态突变异常记录数：{df_abn.shape[0]}")
    if not df_abn.empty:
        top_ids = df_abn['id'].value_counts().head(5)
        log.info(f"异常次数最多的车辆（前5）：\n{top_ids}")

    df = df.loc[~df.index.isin(df_abn['index'].values)].reset_index(drop=True)
    log.info(f"剔除短时间状态突变异常后：{before} -> {df.shape[0]}")

    # 清掉辅助列，避免污染输出文件；OD提取阶段会重新生成需要的shift列
    df = df.drop(columns=['status_up', 'status_down', 'id_up', 'id_down', 'time_up', 'time_down'])
    return df


# --------------------------- 6. OD（上下车点）初步提取（需求11，02阶段任务） ---------------------------
def extract_od(df: pd.DataFrame) -> pd.DataFrame:
    """
    需求11：把GPS明细表转换为出行（OD）信息表。
    逻辑：
        status_chg = status - status_up，1表示上车，-1表示下车
        筛选出同一车辆下，"上车行"与紧随其后的"下车行"，错位拼接为一行订单记录
    """
    df = df.sort_values(by=['id', 'time']).reset_index(drop=True)
    df['status_up'] = df['status'].shift(1)
    df['id_up'] = df['id'].shift(1)

    df['status_chg'] = df['status'] - df['status_up']
    df['id_chg'] = df['id'] - df['id_up']

    df_temp = df.loc[((df['status_chg'] == 1) | (df['status_chg'] == -1)) & (df['id_chg'] == 0)].copy()

    df_temp['Etime'] = df_temp['time'].shift(-1)
    df_temp['Elong'] = df_temp['long'].shift(-1)
    df_temp['Elati'] = df_temp['lati'].shift(-1)
    df_temp['Eid'] = df_temp['id'].shift(-1)

    df_order = df_temp.loc[
        (df_temp['status_chg'] == 1) & (df_temp['id'] == df_temp['Eid']),
        ['id', 'time', 'long', 'lati', 'Etime', 'Elong', 'Elati']
    ].copy()

    df_order.columns = ['车辆id', '开始时间', '开始经度', '开始纬度', '结束时间', '结束经度', '结束纬度']
    log.info(f"OD初步提取完成，订单数：{df_order.shape[0]}")
    return df_order.reset_index(drop=True)


# --------------------------- 7. 主流程 ---------------------------
def main():
    log.info("========== 数据清洗流程开始 ==========")

    df = load_data(RAW_DATA_PATH, use_chunk=USE_CHUNK, chunk_size=CHUNK_SIZE)

    df = basic_clean(df)                      # 需求1-2：排序 + 时间类型转换
    df = drop_invalid_records(df)             # 越界/非法值清洗
    df = drop_always_same_status_vehicles(df) # 全天状态不变的异常车辆

    n_before_dup = df.shape[0]
    df = remove_duplicates(df)                # 需求3-7：复杂重复值清洗
    n_after_dup = df.shape[0]

    n_before_abn = df.shape[0]
    df = remove_abnormal_status(df)           # 需求8-10：短时间状态突变异常清洗
    n_after_abn = df.shape[0]

    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False, encoding="utf-8-sig")
    log.info(f"清洗后数据已保存至 {CLEAN_DATA_PATH}，最终shape={df.shape}")

    df_order = extract_od(df)                 # 需求11：OD初步提取
    df_order.to_csv(OD_DATA_PATH, index=False, encoding="utf-8-sig")
    log.info(f"OD数据已保存至 {OD_DATA_PATH}，订单数={df_order.shape[0]}")

    log.info("========== 检查点汇总 ==========")
    log.info(f"重复值清洗：{n_before_dup} -> {n_after_dup}（剔除 {n_before_dup - n_after_dup}）")
    log.info(f"异常值清洗：{n_before_abn} -> {n_after_abn}（剔除 {n_before_abn - n_after_abn}）")
    log.info(f"字段类型：\n{df.dtypes}")
    log.info(f"上车点数量：{(df_order['车辆id'].notna()).sum()}，下车点已配对（即订单数）：{df_order.shape[0]}")
    log.info("========== 数据清洗流程结束 ==========")


if __name__ == "__main__":
    main()