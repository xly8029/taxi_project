# -*- coding: utf-8 -*-
"""
03 OD完成与缓存构建
======================
依据：03-OD完成与缓存构建.html

本模块衔接 data_cleaning.py 的输出（清洗后数据 taxi_clean.csv），完成：
    1. OD表收尾：显式处理"缺少下车点/时间为负/距离异常"三类异常订单（不再只是打标记，而是单独输出异常表）
    2. 车辆缓存：按车辆ID拆分清洗数据，支持按车辆快速读取轨迹
    3. 分钟缓存：按分钟重采样，支持按某一分钟快速查询所有车辆位置
    4. OD缓存：供热力图、订单统计、距离分析、ETA直接复用

目录约定（和 data_cleaning.py 的 CLEAN_DATA_PATH 等保持一致）：
    data/processed/taxi_clean.csv      清洗后明细数据（输入）
    data/processed/taxi_od.csv         OD出行表（输入，由data_cleaning.py生成）
    data/processed/taxi_od_invalid.csv 异常订单表（本模块输出，用于抽查/复核）
    data/cache/vehicle/{id}.csv        车辆缓存：每辆车一个文件
    data/cache/minute/{date}/{HHMM}.csv 分钟缓存：每分钟一个文件，存"该分钟末尾"每辆车的位置
    data/cache/od/od_cache.csv          OD缓存：供热力图/统计/ETA直接读取（剔除异常订单后的版本）
"""

import os
import glob
import logging
import numpy as np
import pandas as pd

# ========================= 配置区 =========================
CLEAN_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/taxi_clean.csv")
OD_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/taxi_od.csv")
OD_INVALID_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/taxi_od_invalid.csv")

VEHICLE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "../data/cache/vehicle")
MINUTE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "../data/cache/minute")
OD_CACHE_PATH = os.path.join(os.path.dirname(__file__), "../data/cache/od/od_cache.csv")

LOG_PATH = os.path.join(os.path.dirname(__file__), "../docs/cache_log.txt")

# OD异常订单判断阈值（和data_cleaning.py里is_valid的阈值保持一致，避免两边标准不统一）
OD_MAX_DISTANCE_KM = 100
OD_MAX_DURATION_HOUR = 3
OD_MIN_DURATION_SEC = 30

MINUTE_FREQ = "1min"   # 分钟缓存的重采样频率
# ===============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# --------------------------- 1. OD表收尾：异常订单显式处理 ---------------------------
def split_valid_invalid_orders(df_order: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    把 data_cleaning.py 生成的 OD 表，按03阶段文档要求显式拆分为"正常订单"和"异常订单"两张表，
    而不是只停留在 is_valid 标记层面。异常订单单独存一份，方便抽查复核（注意事项里要求的"抽查"）。

    三类异常（对应文档原话）：
        a) 缺少下车点：开始时间或结束时间为空（说明配对没成功，理论上 extract_od 已经保证配对，
           这里再做一次兜底检查，防止上游数据有遗漏）
        b) 时间为负：结束时间早于开始时间（订单时长<0）
        c) 距离明显异常：超过 OD_MAX_DISTANCE_KM，或时长超过 OD_MAX_DURATION_HOUR，
           或时长短于 OD_MIN_DURATION_SEC
    """
    df_order = df_order.copy()
    df_order['开始时间'] = pd.to_datetime(df_order['开始时间'])
    df_order['结束时间'] = pd.to_datetime(df_order['结束时间'])

    if '订单时长_秒' not in df_order.columns:
        df_order['订单时长_秒'] = (df_order['结束时间'] - df_order['开始时间']).dt.total_seconds()

    cond_missing = df_order['开始时间'].isna() | df_order['结束时间'].isna()        #开始 / 结束时间为空
    cond_negative_time = df_order['订单时长_秒'] < 0     #订单时长为负
    cond_too_short = df_order['订单时长_秒'] < OD_MIN_DURATION_SEC
    cond_too_long = df_order['订单时长_秒'] > OD_MAX_DURATION_HOUR * 3600

    if '轨迹距离_km' not in df_order.columns:
        # 没有距离字段时兜底算一个（正常情况下 data_cleaning.py 已经算好了）
        R = 6371.0
        lng1, lat1 = np.radians(df_order['开始经度']), np.radians(df_order['开始纬度'])
        lng2, lat2 = np.radians(df_order['结束经度']), np.radians(df_order['结束纬度'])
        a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
        df_order['轨迹距离_km'] = (R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).round(3)

    cond_too_far = df_order['轨迹距离_km'] > OD_MAX_DISTANCE_KM

    invalid_mask = cond_missing | cond_negative_time | cond_too_short | cond_too_long | cond_too_far

    df_order['异常原因'] = ''
    df_order.loc[cond_missing, '异常原因'] += '缺少下车点;'
    df_order.loc[cond_negative_time, '异常原因'] += '时间为负;'
    df_order.loc[cond_too_short, '异常原因'] += '时长过短;'
    df_order.loc[cond_too_long, '异常原因'] += '时长过长;'
    df_order.loc[cond_too_far, '异常原因'] += '距离过远;'

    df_valid = df_order.loc[~invalid_mask].drop(columns=['异常原因']).reset_index(drop=True)
    df_invalid = df_order.loc[invalid_mask].reset_index(drop=True)

    log.info(
        f"OD异常订单拆分完成：总订单 {df_order.shape[0]} -> "
        f"正常 {df_valid.shape[0]}，异常 {df_invalid.shape[0]}"
    )
    if not df_invalid.empty:
        log.info(f"异常原因分布：\n{df_invalid['异常原因'].value_counts()}")

    return df_valid, df_invalid


# --------------------------- 2. 车辆缓存：按车辆ID拆分清洗数据 ---------------------------
def build_vehicle_cache(df_clean: pd.DataFrame, cache_dir: str = VEHICLE_CACHE_DIR) -> None:
    """
    建立车辆缓存：每辆车一个csv，按time排序，支持"输入车辆id -> 直接读对应文件"的快速查询，
    不用每次都从几千万行的全量清洗数据里筛选。

    文件命名固定为 {id}.csv，路径固定为 cache_dir，后续地图/轨迹查询模块直接按这个规则拼路径读取，
    不需要每次都改。
    """
    os.makedirs(cache_dir, exist_ok=True)
    df_clean = df_clean.copy()
    df_clean['time'] = pd.to_datetime(df_clean['time'])

    n = 0
    for vid, group in df_clean.groupby('id'):
        group = group.sort_values('time').reset_index(drop=True)
        out_path = os.path.join(cache_dir, f"{vid}.csv")
        group.to_csv(out_path, index=False, encoding="utf-8-sig")
        n += 1

    log.info(f"车辆缓存构建完成：{n} 辆车，输出目录 {cache_dir}")


# --------------------------- 3. 分钟缓存：基于车辆缓存重采样 ---------------------------
def build_minute_cache(
    vehicle_cache_dir: str = VEHICLE_CACHE_DIR,
    minute_cache_dir: str = MINUTE_CACHE_DIR,
    freq: str = MINUTE_FREQ
) -> None:
    """
    建立分钟缓存：表示"该分钟末尾"每辆车的位置，支持"输入某一分钟 -> 直接读所有车辆当时位置"的快速查询。

    做法（文档建议）：按车辆读入车辆缓存 -> 设置time为索引 -> resample(freq).last().ffill()
    注意：ffill() 表示用上一条有效GPS状态延续填充，不代表车辆在该分钟真的产生了新采样，
    这里额外保留一列 is_observed 标记该分钟是否有真实GPS点，避免后续误用"凑出来的"位置。

    为了避免"多个进程/多次调用同时追加写同一个分钟文件"的稳定性问题（文档注意事项里提到的坑），
    这里采用"先在内存里把所有车辆的分钟数据汇总好，再按分钟分组一次性写文件"的方式，
    每个分钟文件只会被写一次，不存在并发追加的问题。
    """
    vehicle_files = glob.glob(os.path.join(vehicle_cache_dir, "*.csv"))
    if not vehicle_files:
        log.warning(f"未找到车辆缓存文件，请先运行 build_vehicle_cache。目录：{vehicle_cache_dir}")
        return

    resampled_list = []
    for fp in vehicle_files:
        df_v = pd.read_csv(fp)
        df_v['time'] = pd.to_datetime(df_v['time'])
        df_v = df_v.set_index('time').sort_index()

        # 标记真实GPS点所在的分钟，resample之前先打标记，避免被ffill污染
        df_v['is_observed'] = True

        df_r = df_v.resample(freq).last()       # 取该分钟最后一条真实GPS记录
        df_r['is_observed'] = df_r['is_observed'].fillna(False)
        df_r = df_r.ffill()
        df_r['is_observed'] = df_r['is_observed'].astype(bool)
        df_r = df_r.reset_index()

        # ffill可能在序列最前面留出NaN（车辆第一条记录之前的分钟），这部分没有意义，直接丢弃
        df_r = df_r.dropna(subset=['id'])

        resampled_list.append(df_r)

    df_all = pd.concat(resampled_list, ignore_index=True)
    df_all['date'] = df_all['time'].dt.strftime('%Y-%m-%d')
    df_all['hhmm'] = df_all['time'].dt.strftime('%H-%M')

    os.makedirs(minute_cache_dir, exist_ok=True)
    n = 0
    for (date, hhmm), group in df_all.groupby(['date', 'hhmm']):
        day_dir = os.path.join(minute_cache_dir, date)
        os.makedirs(day_dir, exist_ok=True)
        out_path = os.path.join(day_dir, f"{hhmm}.csv")
        group.drop(columns=['date', 'hhmm']).to_csv(out_path, index=False, encoding="utf-8-sig")
        n += 1

    log.info(f"分钟缓存构建完成：共 {n} 个分钟文件，输出目录 {minute_cache_dir}")


# --------------------------- 4. OD缓存：剔除异常订单后供下游直接使用 ---------------------------
def build_od_cache(df_valid_order: pd.DataFrame, cache_path: str = OD_CACHE_PATH) -> None:
    """
    建立OD缓存：在"正常订单"基础上，固定输出到统一路径，供热力图、订单统计、距离分析、ETA直接读取。
    后续这些模块不需要再重复跑一遍异常订单拆分逻辑，直接读这个文件即可。
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_valid_order.to_csv(cache_path, index=False, encoding="utf-8-sig")
    log.info(f"OD缓存构建完成：{df_valid_order.shape[0]} 条正常订单，输出至 {cache_path}")


# --------------------------- 5. 抽查（注意事项要求的检查点） ---------------------------
def spot_check(
    vehicle_cache_dir: str = VEHICLE_CACHE_DIR,
    minute_cache_dir: str = MINUTE_CACHE_DIR,
    od_cache_path: str = OD_CACHE_PATH
) -> None:
    """
    按文档"注意事项"要求：缓存生成后要抽查一辆车、一个分钟文件、一个OD订单，确认字段和时间都正确。
    这里自动随机抽一份，打印出来，方便你检查点现场直接展示。
    """
    log.info("========== 抽查开始 ==========")

    vehicle_files = glob.glob(os.path.join(vehicle_cache_dir, "*.csv"))
    if vehicle_files:
        fp = np.random.choice(vehicle_files)
        df_v = pd.read_csv(fp)
        log.info(f"[车辆缓存抽查] 文件={fp}，行数={df_v.shape[0]}\n{df_v.head(3)}")
    else:
        log.warning("[车辆缓存抽查] 未找到车辆缓存文件")

    minute_files = glob.glob(os.path.join(minute_cache_dir, "*", "*.csv"))
    if minute_files:
        fp = np.random.choice(minute_files)
        df_m = pd.read_csv(fp)
        log.info(f"[分钟缓存抽查] 文件={fp}，车辆数={df_m.shape[0]}\n{df_m.head(3)}")
    else:
        log.warning("[分钟缓存抽查] 未找到分钟缓存文件")

    if os.path.exists(od_cache_path):
        df_od = pd.read_csv(od_cache_path)
        if not df_od.empty:
            sample = df_od.sample(1)
            log.info(f"[OD缓存抽查] 总订单数={df_od.shape[0]}\n{sample}")
        else:
            log.warning("[OD缓存抽查] OD缓存为空")
    else:
        log.warning(f"[OD缓存抽查] 未找到OD缓存文件 {od_cache_path}")

    log.info("========== 抽查结束 ==========")


# --------------------------- 6. 主流程 ---------------------------
def main():
    log.info("========== 03 OD完成与缓存构建 开始 ==========")

    # 1) OD表收尾：拆分正常/异常订单
    df_order = pd.read_csv(OD_DATA_PATH)
    df_valid, df_invalid = split_valid_invalid_orders(df_order)
    os.makedirs(os.path.dirname(OD_INVALID_PATH), exist_ok=True)
    df_invalid.to_csv(OD_INVALID_PATH, index=False, encoding="utf-8-sig")
    log.info(f"异常订单表已保存至 {OD_INVALID_PATH}")

    # 2) 车辆缓存
    df_clean = pd.read_csv(CLEAN_DATA_PATH)
    build_vehicle_cache(df_clean)

    # 3) 分钟缓存（基于车辆缓存重采样）
    build_minute_cache()

    # 4) OD缓存（基于正常订单）
    build_od_cache(df_valid)

    # 5) 抽查
    spot_check()

    log.info("========== 03 OD完成与缓存构建 结束 ==========")


if __name__ == "__main__":
    main()