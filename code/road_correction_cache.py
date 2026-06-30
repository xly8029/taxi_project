import hashlib
import json
import os
import pickle
from datetime import datetime

import pandas as pd


def normalize_cache_time(value):
    if value is None or value == "":
        return None
    return str(pd.to_datetime(value))


def safe_time_token(value):
    normalized = normalize_cache_time(value)
    if normalized is None:
        return "full"
    return normalized.replace(":", "-").replace(" ", "_")


def cache_mode_token(kwargs):
    return "undirected" if kwargs.get("use_undirected") else "directed"


def is_full_day_range(start_time, end_time):
    start_norm = normalize_cache_time(start_time)
    end_norm = normalize_cache_time(end_time)
    if not start_norm or not end_norm:
        return False

    start_ts = pd.to_datetime(start_norm)
    end_ts = pd.to_datetime(end_norm)
    return (
        start_ts.strftime("%H:%M:%S") == "00:00:00"
        and end_ts == start_ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    )


def get_single_day_window(start_time, end_time):
    if not start_time or not end_time:
        return None
    start_ts = pd.to_datetime(start_time)
    end_ts = pd.to_datetime(end_time)
    if start_ts.normalize() != end_ts.normalize():
        return None
    day_ts = start_ts.normalize()
    return {
        "day": day_ts.strftime("%Y-%m-%d"),
        "full_start": day_ts.strftime("%Y-%m-%d 00:00:00"),
        "full_end": (day_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S"),
    }


def serialize_cache_kwargs(kwargs):
    normalized = {}
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            normalized[key] = value
        else:
            normalized[key] = repr(value)
    return normalized


def build_correction_cache_key(vehicle_id, start_time=None, end_time=None, *, vehicle_cache_file, network_paths,
                               algo_version, kwargs=None):
    if not os.path.exists(vehicle_cache_file):
        raise FileNotFoundError(f"车辆 {vehicle_id} 的缓存文件不存在")

    kwargs = kwargs or {}
    existing_network_paths = [path for path in network_paths if path and os.path.exists(path)]
    network_mtime = max((os.path.getmtime(path) for path in existing_network_paths), default=None)
    payload = {
        "vehicle_id": vehicle_id,
        "vehicle_cache_mtime": os.path.getmtime(vehicle_cache_file),
        "vehicle_cache_size": os.path.getsize(vehicle_cache_file),
        "network_mtime": network_mtime,
        "start_time": normalize_cache_time(start_time),
        "end_time": normalize_cache_time(end_time),
        "kwargs": serialize_cache_kwargs(kwargs),
        "version": algo_version,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def correction_cache_path(cache_dir, vehicle_id):
    return os.path.join(cache_dir, f"{vehicle_id}.pkl")


def build_cache_slice_key(start_time=None, end_time=None):
    return f"{safe_time_token(start_time)}__to__{safe_time_token(end_time)}"


def normalize_cache_store(store):
    if not isinstance(store, dict):
        return {"entries": {}}
    entries = store.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    store["entries"] = entries
    return store


def load_vehicle_cache_store(cache_dir, vehicle_id, logger):
    cache_path = correction_cache_path(cache_dir, vehicle_id)
    if not os.path.exists(cache_path):
        return {"entries": {}}, cache_path
    try:
        with open(cache_path, "rb") as f:
            return normalize_cache_store(pickle.load(f)), cache_path
    except Exception as exc:
        logger.warning("校正缓存读取失败，准备重算 | %s | %s", os.path.basename(cache_path), exc)
        return {"entries": {}}, cache_path


def write_vehicle_cache_store(cache_dir, vehicle_id, store):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = correction_cache_path(cache_dir, vehicle_id)
    with open(cache_path, "wb") as f:
        pickle.dump(normalize_cache_store(store), f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache_path


def load_correction_cache(cache_dir, cache_key, *, vehicle_id, logger):
    if vehicle_id is None:
        return None
    store, _ = load_vehicle_cache_store(cache_dir, vehicle_id, logger)
    return store.get("entries", {}).get(cache_key)


def write_correction_cache(cache_dir, cache_key, result, *, vehicle_id, algo_version, logger):
    try:
        if vehicle_id is None:
            return
        store, _ = load_vehicle_cache_store(cache_dir, vehicle_id, logger)
        store["entries"][cache_key] = result
        store["meta"] = {
            "vehicle_id": vehicle_id,
            "algo_version": algo_version,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_vehicle_cache_store(cache_dir, vehicle_id, store)
    except Exception as exc:
        logger.warning("校正缓存写入失败，不影响本次结果 | %s", exc)


def cache_file_exists(cache_dir, vehicle_id, start_time=None, end_time=None, *, vehicle_cache_file, network_paths,
                      algo_version, logger, kwargs=None):
    cache_key = build_correction_cache_key(
        vehicle_id,
        start_time,
        end_time,
        vehicle_cache_file=vehicle_cache_file,
        network_paths=network_paths,
        algo_version=algo_version,
        kwargs=kwargs,
    )
    store, cache_path = load_vehicle_cache_store(cache_dir, vehicle_id, logger)
    return cache_key in store.get("entries", {}), cache_path, cache_key


def build_full_day_cache_key(vehicle_id, day, *, vehicle_cache_file, network_paths, algo_version, kwargs=None):
    window = get_single_day_window(f"{day} 00:00:00", f"{day} 23:59:59")
    return build_correction_cache_key(
        vehicle_id,
        window["full_start"],
        window["full_end"],
        vehicle_cache_file=vehicle_cache_file,
        network_paths=network_paths,
        algo_version=algo_version,
        kwargs=kwargs,
    )


def get_full_day_result_from_store(store, vehicle_id, day, *, vehicle_cache_file, network_paths, algo_version,
                                   kwargs=None):
    full_key = build_full_day_cache_key(
        vehicle_id,
        day,
        vehicle_cache_file=vehicle_cache_file,
        network_paths=network_paths,
        algo_version=algo_version,
        kwargs=kwargs,
    )
    return store.get("entries", {}).get(full_key), full_key


def slice_timed_pieces_by_range(timed_pieces, start_time=None, end_time=None):
    start_ts = pd.to_datetime(start_time) if start_time else None
    end_ts = pd.to_datetime(end_time) if end_time else None
    selected = []
    for piece in timed_pieces:
        piece_start = pd.to_datetime(piece["start_time"]) if piece.get("start_time") is not None else None
        piece_end = pd.to_datetime(piece["end_time"]) if piece.get("end_time") is not None else None

        if start_ts is not None and piece_end is not None and piece_end < start_ts:
            continue
        if end_ts is not None and piece_start is not None and piece_start > end_ts:
            continue
        selected.append(piece)
    return selected


def slice_debug_segments_by_range(debug_segments, start_time=None, end_time=None):
    start_ts = pd.to_datetime(start_time) if start_time else None
    end_ts = pd.to_datetime(end_time) if end_time else None
    selected = []
    for segment in debug_segments:
        seg_start = pd.to_datetime(segment.get("start_time")) if segment.get("start_time") is not None else None
        seg_end = pd.to_datetime(segment.get("end_time")) if segment.get("end_time") is not None else None

        if start_ts is not None and seg_end is not None and seg_end < start_ts:
            continue
        if end_ts is not None and seg_start is not None and seg_start > end_ts:
            continue
        selected.append(segment)
    return selected


def slice_cached_result_by_time_range(result, df, start_time=None, end_time=None, *, rebuild_coords_fn):
    sliced = dict(result)
    timed_pieces = result.get("timed_pieces") or []
    if timed_pieces:
        selected_pieces = slice_timed_pieces_by_range(timed_pieces, start_time, end_time)
        corrected_coords, corrected_segments = rebuild_coords_fn(selected_pieces)
        sliced["timed_pieces"] = selected_pieces
        sliced["corrected_coords"] = corrected_coords
        sliced["corrected_segments"] = corrected_segments

    sliced_debug = slice_debug_segments_by_range(result.get("debug_segments", []), start_time, end_time)
    sliced["debug_segments"] = sliced_debug
    sliced["original_coords"] = df[["lati", "long"]].values.tolist()
    return sliced


def load_full_day_cache_for_range(cache_dir, vehicle_id, start_time=None, end_time=None, *, vehicle_cache_file,
                                  network_paths, algo_version, logger, kwargs=None):
    if not start_time or not end_time:
        return None, None, None

    start_ts = pd.to_datetime(start_time)
    end_ts = pd.to_datetime(end_time)
    if start_ts.normalize() != end_ts.normalize():
        return None, None, None

    day = start_ts.strftime("%Y-%m-%d")
    store, _ = load_vehicle_cache_store(cache_dir, vehicle_id, logger)
    cached, full_key = get_full_day_result_from_store(
        store,
        vehicle_id,
        day,
        vehicle_cache_file=vehicle_cache_file,
        network_paths=network_paths,
        algo_version=algo_version,
        kwargs=kwargs,
    )
    return cached, day, full_key
