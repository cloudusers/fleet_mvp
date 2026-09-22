# 搜 argmin L。current_gid 锁在队首，不拆。
# 贪心：按 ETA 插一遍。ALNS：拆已派 + 未派进池，按 regret 回插。

from __future__ import annotations

import math
import random
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from fleet_mvp.execute import simulate_solution
from fleet_mvp.loss import assignment_loss, execution_cost, formula_text, solution_cost, solution_loss
from fleet_mvp.models import Driver, DriverPlan, Group, Solution


def _clone(sol: Solution) -> Solution:
    plans = [DriverPlan(driver=p.driver, groups=list(p.groups)) for p in sol.plans]
    return Solution(plans=plans, unassigned=list(sol.unassigned), locked_gids=list(sol.locked_gids))


def empty_solution(drivers: Sequence[Driver]) -> Solution:
    return Solution(plans=[DriverPlan(driver=d, groups=[]) for d in drivers], unassigned=[])


def _remove_group(sol: Solution, gid: str) -> Optional[Group]:
    if gid in sol.locked_gids:
        return None
    for plan in sol.plans:
        for i, group in enumerate(plan.groups):
            if group.gid == gid:
                return plan.groups.pop(i)
    for i, group in enumerate(sol.unassigned):
        if group.gid == gid:
            return sol.unassigned.pop(i)
    return None


def _min_pos(plan: DriverPlan, locked: Sequence[str]) -> int:
    # 锁定组占位 0，后面的单从 1 插
    if plan.groups and plan.groups[0].gid in locked:
        return 1
    return 0


def try_insert(sol: Solution, group: Group, now: Optional[datetime] = None) -> bool:
    best: Optional[Tuple[float, Solution]] = None
    locked = sol.locked_gids
    for p_idx, plan in enumerate(sol.plans):
        if not plan.driver.consider_for(group):
            continue
        lo = _min_pos(plan, locked)
        for pos in range(lo, len(plan.groups) + 1):
            cand = _clone(sol)
            cand.plans[p_idx].groups.insert(pos, group)
            cost = solution_cost(cand, now=now)
            if cost >= 1e12:
                continue
            if best is None or cost < best[0]:
                best = (cost, cand)
    if best is None:
        sol.unassigned.append(group)
        return False
    sol.plans = best[1].plans
    sol.unassigned = best[1].unassigned
    return True


def _seed_locked(drivers: Sequence[Driver], groups: Sequence[Group]) -> Tuple[Solution, List[Group]]:
    by_id = {g.gid: g for g in groups}
    sol = empty_solution(drivers)
    locked = []
    used = set()
    for plan in sol.plans:
        gid = plan.driver.current_gid
        if gid and gid in by_id:
            plan.groups.append(by_id[gid])
            locked.append(gid)
            used.add(gid)
    sol.locked_gids = locked
    rest = [g for g in groups if g.gid not in used]
    return sol, rest


def greedy_assign(drivers: Sequence[Driver], groups: Sequence[Group], now: Optional[datetime] = None) -> Solution:
    sol, rest = _seed_locked(drivers, groups)
    for group in sorted(rest, key=lambda g: (g.first.eta, g.gid)):
        try_insert(sol, group, now=now)
    return sol


def _feasible_insert_costs(sol: Solution, group: Group, now: Optional[datetime] = None) -> List[float]:
    # 每个能接的司机，把这组插进去后的方案 L（取该司机最好位置）
    costs: List[float] = []
    locked = sol.locked_gids
    for p_idx, plan in enumerate(sol.plans):
        if not plan.driver.consider_for(group):
            continue
        best = 1e12
        found = False
        lo = _min_pos(plan, locked)
        for pos in range(lo, len(plan.groups) + 1):
            cand = _clone(sol)
            cand.plans[p_idx].groups.insert(pos, group)
            cost = solution_cost(cand, now=now)
            if cost < 1e12 and cost < best:
                best = cost
                found = True
        if found:
            costs.append(best)
    costs.sort()
    return costs


class GroupALNS:
    def __init__(
        self,
        drivers: Sequence[Driver],
        groups: Sequence[Group],
        max_iter: int = 80,
        seed: int = 42,
        now: Optional[datetime] = None,
    ):
        self.drivers = list(drivers)
        self.groups = list(groups)
        self.max_iter = max_iter
        self.rng = random.Random(seed)
        self.now = now
        self.destroy_ops = [self.destroy_random, self.destroy_worst, self.destroy_related]
        self.w_destroy = [1.0] * 3
        self.temp = 60.0
        self.cool = 0.97

    def _movable(self, sol: Solution) -> List[Group]:
        locked = set(sol.locked_gids)
        return [g for g in sol.assigned_groups() if g.gid not in locked]

    def destroy_random(self, sol: Solution) -> List[Group]:
        assigned = self._movable(sol)
        if not assigned:
            return []
        k = max(1, min(len(assigned), int(len(self.groups) * 0.25) or 1))
        picked = self.rng.sample(assigned, k)
        return [g for g in (_remove_group(sol, p.gid) for p in picked) if g]

    def destroy_worst(self, sol: Solution) -> List[Group]:
        sim = simulate_solution(sol, now=self.now)
        scored = []
        locked = set(sol.locked_gids)
        if sim.feasible:
            for plan, ds in zip(sol.plans, sim.driver_sims):
                for group, exe in zip(plan.groups, ds.executions):
                    if group.gid in locked:
                        continue
                    scored.append((execution_cost(exe), group.gid))
        if not scored:
            return self.destroy_random(sol)
        scored.sort(reverse=True)
        k = max(1, min(len(scored), int(len(self.groups) * 0.2) or 1))
        removed = []
        for _, gid in scored[:k]:
            found = _remove_group(sol, gid)
            if found:
                removed.append(found)
        return removed

    def destroy_related(self, sol: Solution) -> List[Group]:
        assigned = self._movable(sol)
        if not assigned:
            return []
        seed = self.rng.choice(assigned)
        related = [
            g for g in assigned if abs((g.first.eta - seed.first.eta).total_seconds()) <= 2400
        ]
        if len(related) < 2:
            related = assigned
        k = max(1, min(len(related), int(len(self.groups) * 0.22) or 1))
        picked = self.rng.sample(related, k)
        return [g for g in (_remove_group(sol, p.gid) for p in picked) if g]

    def roulette(self, weights: List[float]) -> int:
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                return i
        return len(weights) - 1

    def _repair(self, sol: Solution, groups: Sequence[Group]) -> None:
        # 未派也进池。优先插「只有一个人能接」的组，避免再按 ETA 占死人。
        remaining = list(groups)
        while remaining:
            pick = remaining[0]
            pick_key = None
            for group in remaining:
                costs = _feasible_insert_costs(sol, group, self.now)
                if not costs:
                    regret = -1.0
                elif len(costs) == 1:
                    regret = 1e12
                else:
                    regret = costs[1] - costs[0]
                key = (regret, group.people(), group.first.eta, group.gid)
                if pick_key is None or key > pick_key:
                    pick, pick_key = group, key
            remaining.remove(pick)
            try_insert(sol, pick, now=self.now)

    def run(self, verbose: bool = True) -> Solution:
        current = greedy_assign(self.drivers, self.groups, now=self.now)
        best = _clone(current)
        best_cost = solution_cost(best, now=self.now)
        curr_cost = best_cost
        if verbose:
            print(
                f"初始贪心：已派 {len(best.assigned_groups())}/{len(self.groups)}，"
                f"L={best_cost:.1f}，未派 {[g.gid for g in best.unassigned]}"
            )
        for it in range(self.max_iter):
            cand = _clone(current)
            removed = self.destroy_ops[self.roulette(self.w_destroy)](cand)
            pool = removed + list(cand.unassigned)
            cand.unassigned = []
            self._repair(cand, pool)
            new_cost = solution_cost(cand, now=self.now)
            if new_cost >= 1e12:
                self.temp *= self.cool
                continue
            accept = new_cost <= curr_cost
            if not accept:
                delta = new_cost - curr_cost
                accept = math.exp(-delta / max(self.temp, 1e-8)) > self.rng.random()
            if accept:
                current = cand
                curr_cost = new_cost
                if new_cost + 1e-6 < best_cost:
                    best = _clone(cand)
                    best_cost = new_cost
                    if verbose:
                        print(
                            f"迭代{it} 更优：已派 {len(best.assigned_groups())}/{len(self.groups)}，"
                            f"L={best_cost:.1f}，未派 {[g.gid for g in best.unassigned]}"
                        )
            self.temp *= self.cool
        return best


def rank_candidates(sol: Solution, group: Group, now: Optional[datetime] = None, top: int = 3):
    base = _clone(sol)
    _remove_group(base, group.gid)
    base.unassigned = [g for g in base.unassigned if g.gid != group.gid]
    rows = []
    locked = base.locked_gids
    for p_idx, plan in enumerate(base.plans):
        if not plan.driver.consider_for(group):
            rows.append((plan.driver.did, 1e12, "下班或不在班次内"))
            continue
        best_cost = 1e12
        best_why = "插不进"
        lo = _min_pos(plan, locked)
        for pos in range(lo, len(plan.groups) + 1):
            cand = _clone(base)
            cand.plans[p_idx].groups.insert(pos, group)
            sim = simulate_solution(cand, now=now)
            if not sim.feasible:
                continue
            exe = None
            for g, e in zip(cand.plans[p_idx].groups, sim.driver_sims[p_idx].executions):
                if g.gid == group.gid:
                    exe = e
                    break
            if exe is None:
                continue
            part = assignment_loss(exe)
            if part.total < best_cost:
                best_cost = part.total
                best_why = part.explain()
        rows.append((plan.driver.did, best_cost, best_why))
    rows.sort(key=lambda x: x[1])
    return rows[:top]


def _period_label(group: Group) -> str:
    hour = group.first.eta.hour
    if hour < 12:
        return "早晨"
    if hour < 17:
        return "白天"
    return "晚上"


def chain_transition_stats(sol: Solution, sim) -> List[str]:
    buckets = {
        "送机→送机": [],
        "送机→接机": [],
        "接机→接机": [],
        "接机→送机": [],
    }
    for plan, ds in zip(sol.plans, sim.driver_sims):
        for prev, nxt, exe in zip(plan.groups, plan.groups[1:], ds.executions[1:]):
            key = f"{prev.kind_label()}→{nxt.kind_label()}"
            buckets.setdefault(key, []).append(exe.empty_km)
    lines = ["接龙空驶："]
    for key, kms in buckets.items():
        if not kms:
            lines.append(f"  {key}  未出现")
            continue
        avg = sum(kms) / len(kms)
        lines.append(f"  {key}  {len(kms)} 次  空驶合计 {sum(kms):.1f} km  均 {avg:.1f} km")
    return lines


def format_dispatch(sol: Solution, now: Optional[datetime] = None) -> str:
    sim = simulate_solution(sol, now=now, policy="jit")
    total = solution_loss(sol, now=now)
    n_songji = sum(1 for g in sol.assigned_groups() if g.is_to_station())
    n_jieji = sum(1 for g in sol.assigned_groups() if not g.is_to_station())
    lines = [
        "======== 派单方案 ========",
        formula_text(),
        f"本方案 {total.explain() if not total.infeasible else 'L=∞'}",
        f"空驶合计 {total.empty_km:.1f} km / {total.empty_min:.0f} min  已派送机 {n_songji} / 接机 {n_jieji}",
    ]
    if not sim.feasible:
        lines.append(f"方案不可行：{sim.reason}")
        return "\n".join(lines)
    lines.extend(chain_transition_stats(sol, sim))

    for plan, ds in zip(sol.plans, sim.driver_sims):
        d = plan.driver
        head = f"{d.did} {d.name} {d.plate}  GPS({d.lat:.4f},{d.lon:.4f})  {d.status_label()}"
        if not plan.groups:
            lines.append(f"{head}  待命")
            continue
        lines.append(f"{head}  | {len(plan.groups)} 组  空驶{ds.empty_km:.1f}km")
        for group, exe in zip(plan.groups, ds.executions):
            lock = "【当前在跑】" if group.gid in sol.locked_gids else ""
            ot = ""
            if exe.overtime:
                ot = (
                    f"【加班】预计 {exe.end_at.strftime('%H:%M')} 收车，"
                    f"下班 {plan.driver.shift_end.strftime('%H:%M')}，超时 {exe.overtime_min:.0f} 分钟"
                )
            part = assignment_loss(exe)
            eta_name = "接客" if group.is_to_station() else "送达"
            lines.append(
                f"  {group.gid} {_period_label(group)}{group.kind_label()} {group.route_text()} {lock}{ot} "
                f"{group.note}  {group.people()}人  {part.explain()}"
            )
            if group.gid in sol.locked_gids:
                lines.append(
                    f"    本趟进行中，{d.free_at.strftime('%H:%M')} 到达终点后接下单；"
                    f"终点 ({exe.end_lat:.4f},{exe.end_lon:.4f})"
                )
            else:
                lines.append(
                    f"    建议派单 {exe.dispatch_at.strftime('%H:%M')}  "
                    f"出发 {exe.depart.strftime('%H:%M')}  "
                    f"最晚 {exe.latest_depart.strftime('%H:%M')}  "
                    f"首点{eta_name} {exe.arrive_first.strftime('%H:%M')} / 预估 {group.first.eta.strftime('%H:%M')}  "
                    f"空驶{exe.empty_km:.1f}km"
                )

    if sol.unassigned:
        lines.append("未派分组：")
        for g in sol.unassigned:
            lines.append(
                f"  {g.gid} {g.kind_label()} {g.route_text()}  "
                f"首点预估 {g.first.eta.strftime('%H:%M')}  {g.people()}人"
            )
    else:
        lines.append("全部组已派发。")
    return "\n".join(lines)
