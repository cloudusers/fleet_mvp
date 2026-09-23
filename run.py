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
from fleet_mvp.compare_result import write_real_comparison
from fleet_mvp.export_result import allocate_result_stem, write_dispatch_xlsx
from fleet_mvp.io_util import default_dataset


def _parse_dt(value: str | None):
    if not value:
        return None
    return datetime.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="接送机编组派单")
    parser.add_argument("--peak", action="store_true", help="用 groups_peak.json")
    parser.add_argument("--trap", action="store_true", help="贪心会占死人、ALNS 能救回来的对照数据")
    parser.add_argument("--real", action="store_true", help="用 2026-09-22 线上配车生成的数据")
    parser.add_argument("--start", default=None, help="当前时刻，默认最早 free_at")
    parser.add_argument("--alns", action="store_true", help="贪心后再跑 ALNS")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if sum(bool(flag) for flag in (args.peak, args.trap, args.real)) > 1:
        parser.error("peak、trap、real 只能选一个")

    drivers, groups = default_dataset(peak=args.peak, trap=args.trap, real=args.real)
    now = _parse_dt(args.start)
    if now is None and drivers:
        now = min(d.free_at for d in drivers)

    n_songji = sum(1 for g in groups if g.is_to_station())
    n_jieji = len(groups) - n_songji
    searcher = "ALNS" if args.alns else "贪心"
    print(f"司机 {len(drivers)}  编组 {len(groups)}（送机 {n_songji} / 接机 {n_jieji}）  搜索：{searcher}")

    greedy_sol = greedy_assign(drivers, groups, now=now) if args.real else None
    if args.alns:
        sol = GroupALNS(drivers, groups, max_iter=args.iters, seed=args.seed, now=now).run(verbose=True)
        print()
    elif greedy_sol is not None:
        sol = greedy_sol
    else:
        sol = greedy_assign(drivers, groups, now=now)
    print(format_dispatch(sol, now=now), flush=True)
    if args.real:
        stem = allocate_result_stem()
        saved = write_dispatch_xlsx(sol, now=now, path=stem.with_suffix(".xlsx"))
        if args.alns:
            alns_sol = sol
        else:
            print(f"对比文档再跑 ALNS {args.iters} 轮", flush=True)
            alns_sol = GroupALNS(drivers, groups, max_iter=args.iters, seed=args.seed, now=now).run(verbose=False)
        compared = write_real_comparison(
            stem.with_suffix(".md"),
            drivers,
            groups,
            greedy_sol,
            alns_sol,
            now,
            iters=args.iters,
            seed=args.seed,
            xlsx_name=saved.name,
            xlsx_mode="ALNS" if args.alns else "贪心",
        )
        print(f"运力表 {saved}", flush=True)
        print(f"对比 {compared}", flush=True)
    else:
        saved = write_dispatch_xlsx(sol, now=now)
        print(f"运力表 {saved}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
