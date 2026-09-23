# 线上原派、贪心、ALNS 的对比。只在 --real 时写，文件名和本次运力表相同。

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from fleet_mvp.config import AVERAGE_SPEED_KMH
from fleet_mvp.execute import execute_group, simulate_solution
from fleet_mvp.loss import solution_loss
from fleet_mvp.models import Driver, Group, Solution

_HISTORY = re.compile(r"历史司机(.+?)(?:【([^】]*)】)?$")


def _row(group: Group, name: str, plate: str, exe) -> dict:
    span = (group.last.eta - group.first.eta).total_seconds() / 60.0
    return {
        "gid": group.gid,
        "driver": name,
        "plate": plate or "",
        "kind": group.kind_label(),
        "people": group.people(),
        "guest": group.first.guest or group.first.name or group.first.oid,
        "span_min": span,
        "empty_km": exe.empty_km,
        "empty_min": exe.empty_min,
        "loaded_km": exe.trip_km - exe.empty_km,
        "delay": exe.eta_delay_min,
        "depart": exe.depart,
        "end": exe.end_at,
    }


def replay_history(
    drivers: Sequence[Driver], groups: Sequence[Group], now: Optional[datetime]
) -> Tuple[Dict[str, dict], Dict[str, int], List[Group]]:
    # 按历史司机、首站预估时刻排序。接不上就跳过，人留在上一单终点。
    by_name = {driver.name: driver for driver in drivers}
    bucket: dict[str, List[Tuple[Group, str]]] = defaultdict(list)
    for group in groups:
        matched = _HISTORY.search(group.note or "")
        if not matched:
            raise ValueError(f"{group.gid} 没有历史司机")
        name = matched.group(1)
        if name not in by_name:
            raise ValueError(f"{group.gid} 的历史司机 {name} 不在司机列表里")
        bucket[name].append((group, matched.group(2) or ""))
    done: Dict[str, dict] = {}
    counts = {driver.name: 0 for driver in drivers}
    skipped: List[Group] = []
    for name, items in bucket.items():
        driver = by_name[name]
        lat, lon, free_at = driver.dispatch_origin()
        for group, plate in sorted(items, key=lambda item: (item[0].first.eta, item[0].gid)):
            exe = execute_group(
                group, lat, lon, free_at, driver=driver, now=now, policy="jit", need_latest=True
            )
            if not exe.feasible or exe.end_at is None:
                skipped.append(group)
                continue
            done[group.gid] = _row(group, name, plate or driver.plate, exe)
            counts[name] += 1
            lat, lon, free_at = exe.end_lat, exe.end_lon, exe.end_at
    return done, counts, skipped


def solution_rows(
    sol: Solution, now: Optional[datetime]
) -> Tuple[Dict[str, dict], Dict[str, int], float]:
    sim = simulate_solution(sol, now=now, policy="jit")
    if not sim.feasible:
        raise ValueError(sim.reason)
    rows: Dict[str, dict] = {}
    counts: Dict[str, int] = {}
    for plan, driver_sim in zip(sol.plans, sim.driver_sims):
        counts[plan.driver.name] = len(plan.groups)
        for group, exe in zip(plan.groups, driver_sim.executions):
            rows[group.gid] = _row(group, plan.driver.name, plan.driver.plate, exe)
    for driver_name in (plan.driver.name for plan in sol.plans):
        counts.setdefault(driver_name, 0)
    loss = solution_loss(sol, now=now)
    return rows, counts, loss.total


def _totals(rows: Dict[str, dict]) -> dict:
    empty_km = sum(row["empty_km"] for row in rows.values())
    empty_min = sum(row["empty_min"] for row in rows.values())
    loaded_km = sum(row["loaded_km"] for row in rows.values())
    people = sum(row["people"] for row in rows.values())
    delay = sum(row["delay"] for row in rows.values())
    denom = loaded_km + empty_km
    return {
        "trips": len(rows),
        "people": people,
        "empty_km": empty_km,
        "empty_min": empty_min,
        "loaded_km": loaded_km,
        "delay": delay,
        "util": 100.0 * loaded_km / denom if denom else 0.0,
    }


def _load_span(counts: Dict[str, int], names: Sequence[str]) -> str:
    vals = sorted(counts.get(name, 0) for name in names)
    if not vals:
        return "—"
    mid = len(vals) // 2
    if len(vals) % 2:
        median = float(vals[mid])
    else:
        median = (vals[mid - 1] + vals[mid]) / 2
    median_text = f"{median:.0f}" if median == int(median) else f"{median:.1f}"
    return f"{vals[0]}–{vals[-1]}，中位 {median_text}"


def _km(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return f"{round(value):.0f} km"
    return f"{value:.1f} km"


def _minutes(value: float) -> str:
    return f"{round(value):.0f} 分钟"


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _overlaps(rows: Dict[str, dict]) -> int:
    by_plate: dict[str, List[dict]] = defaultdict(list)
    for row in rows.values():
        if not row["plate"] or row["plate"] == "无车号" or row["depart"] is None or row["end"] is None:
            continue
        by_plate[row["plate"]].append(row)
    pairs = set()
    for plate, items in by_plate.items():
        items = sorted(items, key=lambda row: row["depart"])
        for index, left in enumerate(items):
            for right in items[index + 1 :]:
                if right["depart"] >= left["end"]:
                    break
                if left["driver"] == right["driver"]:
                    continue
                pairs.add((plate, tuple(sorted((left["driver"], right["driver"])))))
    return len(pairs)


def _changed_examples(greedy: Dict[str, dict], alns: Dict[str, dict]) -> Tuple[int, int, int, int, List[list], List[list]]:
    both = set(greedy) & set(alns)
    changed = [gid for gid in both if greedy[gid]["driver"] != alns[gid]["driver"]]
    better = worse = tie = 0
    scored = []
    for gid in changed:
        delta = alns[gid]["empty_km"] - greedy[gid]["empty_km"]
        if abs(delta) < 0.05:
            tie += 1
        elif delta < 0:
            better += 1
        else:
            worse += 1
        if greedy[gid]["span_min"] <= 90:
            scored.append((delta, gid))
    scored.sort(key=lambda item: item[0])
    wins = [_example(greedy, alns, gid) for delta, gid in scored if delta < -0.4][:3]
    losses = [_example(greedy, alns, gid) for delta, gid in reversed(scored) if delta > 0.4][:2]
    return len(changed), better, worse, tie, wins, losses


def _example(greedy: Dict[str, dict], alns: Dict[str, dict], gid: str) -> list:
    left, right = greedy[gid], alns[gid]
    return [
        gid,
        f"{left['guest']} {left['kind']}",
        left["driver"],
        right["driver"],
        f"{left['empty_km']:.1f} → {right['empty_km']:.1f} km",
    ]


def render_comparison(
    drivers: Sequence[Driver],
    groups: Sequence[Group],
    greedy: Solution,
    alns: Solution,
    now: Optional[datetime],
    *,
    iters: int,
    seed: int,
    xlsx_name: str,
    xlsx_mode: str,
) -> str:
    hist, hist_counts, skipped = replay_history(drivers, groups, now)
    g_rows, g_counts, g_loss = solution_rows(greedy, now)
    a_rows, a_counts, a_loss = solution_rows(alns, now)
    names = [driver.name for driver in drivers]
    h_all, g_all, a_all = _totals(hist), _totals(g_rows), _totals(a_rows)
    common = set(hist) & set(g_rows) & set(a_rows)
    h_same = _totals({gid: hist[gid] for gid in common})
    g_same = _totals({gid: g_rows[gid] for gid in common})
    a_same = _totals({gid: a_rows[gid] for gid in common})
    same_g = sum(1 for gid, row in g_rows.items() if gid in hist and row["driver"] == hist[gid]["driver"])
    same_a = sum(1 for gid, row in a_rows.items() if gid in hist and row["driver"] == hist[gid]["driver"])
    changed, better, worse, tie, wins, losses = _changed_examples(g_rows, a_rows)
    order = sorted(names, key=lambda name: (-g_counts.get(name, 0), name))
    g_miss = [group.gid for group in greedy.unassigned]
    a_miss = [group.gid for group in alns.unassigned]

    lines = [
        "# 0922：线上、贪心、ALNS",
        "",
        (
            f"同一次运行写出 `{xlsx_name}`，这份表格是 {xlsx_mode}。"
            f"对比里的 ALNS 是 {iters} 轮、种子 {seed}。"
            f"空驶是上一单终点到下一单载客点，直线 {AVERAGE_SPEED_KMH:.0f} km/h。"
            "线上按历史司机、首站预估时刻排序，接不上就跳过，人留在上一单终点。"
        ),
        "",
        (
            f"里程利用率 {_pct(h_all['util'])} → {_pct(g_all['util'])} → {_pct(a_all['util'])}。"
            f"全天空驶 {_km(h_all['empty_km'])} → {_km(g_all['empty_km'])} → {_km(a_all['empty_km'])}。"
            f"跑成 {h_all['trips']} → {g_all['trips']} → {a_all['trips']} 趟。"
            f"单人最多 {max(hist_counts.values() or [0])} → {max(g_counts.values() or [0])} → {max(a_counts.values() or [0])} 趟。"
        ),
        "",
        _verdict(g_all, a_all, g_loss, a_loss, g_miss, a_miss, greedy, alns),
        "",
        f"## 和线上比：同一 {len(common)} 趟",
        "",
        f"这 {len(common)} 趟线上原司机跑得完，贪心和 ALNS 也都派了。载客里程都是 {_km(h_same['loaded_km'])}。",
        "",
        _table(
            ["", "线上原派", "贪心", "ALNS"],
            [
                ["空驶距离", _km(h_same["empty_km"]), _km(g_same["empty_km"]), _km(a_same["empty_km"])],
                ["空驶时间", _minutes(h_same["empty_min"]), _minutes(g_same["empty_min"]), _minutes(a_same["empty_min"])],
                ["里程利用率", _pct(h_same["util"]), _pct(g_same["util"]), _pct(a_same["util"])],
                ["晚点", _minutes(h_same["delay"]), _minutes(g_same["delay"]), _minutes(a_same["delay"])],
                ["和线上同一司机的单", "—", f"{same_g} 张", f"{same_a} 张"],
            ],
        ),
        "",
        "## 全天",
        "",
        _table(
            ["", "线上原派", "贪心", "ALNS"],
            [
                [
                    "跑成 / 未派",
                    f"{h_all['trips']} / 跳过 {len(skipped)}",
                    f"{g_all['trips']} / {len(g_miss)}",
                    f"{a_all['trips']} / {len(a_miss)}",
                ],
                ["运送人数", str(h_all["people"]), str(g_all["people"]), str(a_all["people"])],
                [
                    "空驶",
                    f"{_km(h_all['empty_km'])} / {_minutes(h_all['empty_min'])}",
                    f"{_km(g_all['empty_km'])} / {_minutes(g_all['empty_min'])}",
                    f"{_km(a_all['empty_km'])} / {_minutes(a_all['empty_min'])}",
                ],
                ["载客里程", _km(h_all["loaded_km"]), _km(g_all["loaded_km"]), _km(a_all["loaded_km"])],
                ["里程利用率", _pct(h_all["util"]), _pct(g_all["util"]), _pct(a_all["util"])],
                ["总账 L", "—", f"{g_loss:.0f}", f"{a_loss:.0f}"],
                [
                    "每人趟数",
                    _load_span(hist_counts, names),
                    _load_span(g_counts, names),
                    _load_span(a_counts, names),
                ],
            ],
        ),
        "",
        (
            f"主车号时段相交：线上 {_overlaps(hist)} 对，贪心 {_overlaps(g_rows)} 对，ALNS {_overlaps(a_rows)} 对。"
            "模型按人派，车号不互斥。"
        ),
        "",
        "## 贪心和 ALNS",
        "",
        (
            f"两边都派成的有 {len(set(g_rows) & set(a_rows))} 张，其中 {changed} 张换了司机。"
            f"按单张空驶，换司机之后更短 {better} 张、更长 {worse} 张、差不多 {tie} 张。"
            f"{_empty_gap(g_all['empty_km'], a_all['empty_km'])}"
        ),
        "",
    ]
    if wins:
        lines.extend(
            [
                "空驶变短的几张：",
                "",
                _table(["配车单", "客人", "贪心", "ALNS", "空驶"], wins),
                "",
            ]
        )
    if losses:
        lines.extend(
            [
                "空驶变长的几张：",
                "",
                _table(["配车单", "客人", "贪心", "ALNS", "空驶"], losses),
                "",
            ]
        )
    if g_miss or a_miss:
        lines.extend(["未派：", ""])
        lines.extend(_miss_lines(greedy, "贪心"))
        lines.extend(_miss_lines(alns, "ALNS"))
        lines.append("")
    lines.extend(
        [
            "## 每人运送数",
            "",
            "按贪心趟数从多到少。",
            "",
            _table(
                ["司机", "线上", "贪心", "ALNS"],
                [[name, str(hist_counts.get(name, 0)), str(g_counts.get(name, 0)), str(a_counts.get(name, 0))] for name in order],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _empty_gap(greedy_km: float, alns_km: float) -> str:
    gap = greedy_km - alns_km
    if gap > 0.05:
        return f"全天 ALNS 空驶少 {_km(gap)}。"
    if gap < -0.05:
        return f"全天贪心空驶少 {_km(-gap)}。"
    return "全天空驶差不多。"


def _verdict(g_all, a_all, g_loss, a_loss, g_miss, a_miss, greedy: Solution, alns: Solution) -> str:
    same_miss = set(g_miss) == set(a_miss)
    if g_all["trips"] == a_all["trips"] and same_miss:
        return (
            f"ALNS 没有多派成单，未派仍是 {len(a_miss)} 张。"
            f"总账 L 从 {g_loss:.0f} 到 {a_loss:.0f}。"
        )
    return (
        f"贪心跑成 {g_all['trips']} 趟、未派 {len(greedy.unassigned)} 张；"
        f"ALNS 跑成 {a_all['trips']} 趟、未派 {len(alns.unassigned)} 张。"
        f"总账 L 从 {g_loss:.0f} 到 {a_loss:.0f}。"
    )


def _miss_lines(sol: Solution, label: str) -> List[str]:
    if not sol.unassigned:
        return [f"- {label}：没有未派。"]
    lines = []
    for group in sol.unassigned:
        guest = "、".join(stop.guest or stop.name or stop.oid for stop in group.stops)
        lines.append(f"- {label} {group.gid} {group.kind_label()} {guest}")
    return lines


def write_real_comparison(
    path: Path,
    drivers: Sequence[Driver],
    groups: Sequence[Group],
    greedy: Solution,
    alns: Solution,
    now: Optional[datetime],
    *,
    iters: int,
    seed: int,
    xlsx_name: str,
    xlsx_mode: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_comparison(
        drivers,
        groups,
        greedy,
        alns,
        now,
        iters=iters,
        seed=seed,
        xlsx_name=xlsx_name,
        xlsx_mode=xlsx_mode,
    )
    path.write_text(text, encoding="utf-8")
    return path
