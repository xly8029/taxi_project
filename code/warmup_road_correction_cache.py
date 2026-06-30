import json
import sys

from road_correction import warmup_vehicle_correction_cache, warmup_all_vehicle_day_caches


def _usage():
    return (
        "用法:\n"
        "  python code/warmup_road_correction_cache.py warmup-day <vehicle_id> <YYYY-MM-DD> [--use-undirected]\n"
        "  python code/warmup_road_correction_cache.py warmup-all-day <YYYY-MM-DD> [--use-undirected] [--limit N] [--force]"
    )


def _parse_args(argv):
    if len(argv) >= 2 and argv[1] == "warmup-day":
        if len(argv) < 4:
            raise ValueError(_usage())
        return {
            "command": "warmup-day",
            "vehicle_id": int(argv[2]),
            "day": argv[3],
            "use_undirected": "--use-undirected" in argv[4:],
        }

    if len(argv) >= 3 and argv[1] == "warmup-all-day":
        limit = None
        if "--limit" in argv[3:]:
            limit_idx = argv.index("--limit")
            if limit_idx + 1 >= len(argv):
                raise ValueError("--limit 后必须跟数量")
            limit = int(argv[limit_idx + 1])
        return {
            "command": "warmup-all-day",
            "day": argv[2],
            "use_undirected": "--use-undirected" in argv[3:],
            "limit": limit,
            "skip_existing": "--force" not in argv[3:],
        }

    raise ValueError(_usage())


if __name__ == "__main__":
    args = _parse_args(sys.argv)
    if args["command"] == "warmup-day":
        result = warmup_vehicle_correction_cache(
            args["vehicle_id"],
            args["day"],
            use_undirected=args["use_undirected"],
        )
    else:
        result = warmup_all_vehicle_day_caches(
            args["day"],
            limit=args["limit"],
            skip_existing=args["skip_existing"],
            use_undirected=args["use_undirected"],
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
