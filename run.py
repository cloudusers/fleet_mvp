# 入口：默认贪心，--alns 再搜一轮。

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fleet_mvp.assign import GroupALNS, format_dispatch, greedy_assign
from fleet_mvp.io_util import default_dataset


def _parse_dt(value: str | None):
    if not value:
        return None
    return datetime.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="接送机编组派单")
    parser.add_argument("--peak", action="store_true", help="用 groups_peak.json")
    parser.add_argument("--trap", action="store_true", help="贪心会占死人、ALNS 能救回来的对照数据")
    parser.add_argument("--start", default=None, help="当前时刻，默认最早 free_at")
    parser.add_argument("--alns", action="store_true", help="贪心后再跑 ALNS")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    drivers, groups = default_dataset(peak=args.peak, trap=args.trap)
    now = _parse_dt(args.start)
    if now is None and drivers:
        now = min(d.free_at for d in drivers)

    n_songji = sum(1 for g in groups if g.is_to_station())
    n_jieji = len(groups) - n_songji
    searcher = "ALNS" if args.alns else "贪心"
    print(f"司机 {len(drivers)}  编组 {len(groups)}（送机 {n_songji} / 接机 {n_jieji}）  搜索：{searcher}")

    if args.alns:
        sol = GroupALNS(drivers, groups, max_iter=args.iters, seed=args.seed, now=now).run(verbose=True)
        print()
    else:
        sol = greedy_assign(drivers, groups, now=now)
    print(format_dispatch(sol, now=now))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
