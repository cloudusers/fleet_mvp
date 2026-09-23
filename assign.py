# 搜 argmin L。current_gid 锁在队首，不拆。
# 贪心：按 ETA 插一遍。ALNS：拆已派 + 未派进池，按 regret 回插。
# 搜索只重算被改到的那条链，不算最晚出发。破坏算子权重按得分更新。

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from fleet_mvp.execute import execute_group, simulate_driver_plan, simulate_solution
from fleet_mvp.loss import (
    DEFAULT_WEIGHTS,
    LossWeights,
    assignment_loss,
    execution_cost,
    solution_loss,
)
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


@dataclass
class _DriverEval:
    feasible: bool
    costs: List[float]
    origins: List[Tuple[float, float, datetime]]

    @property
    def cost(self) -> float:
        return sum(self.costs)


def _miss_penalty(groups: Sequence[Group], weights: LossWeights) -> float:
    people = sum(g.people() for g in groups)
    return weights.unassigned_group * len(groups) + weights.unassigned_person * people


def _eval_driver(
    plan: DriverPlan,
    locked: Sequence[str],
    now: Optional[datetime],
    weights: LossWeights,
) -> _DriverEval:
    ds = simulate_driver_plan(plan, now=now, policy="jit", locked_gids=locked, need_latest=False)
    lat, lon, t = plan.driver.dispatch_origin()
    origins: List[Tuple[float, float, datetime]] = [(lat, lon, t)]
    if not ds.feasible:
        return _DriverEval(False, [], origins)
    costs: List[float] = []
    for exe in ds.executions:
        if exe.end_at is None:
            return _DriverEval(False, [], origins)
        costs.append(execution_cost(exe, weights))
        origins.append((exe.end_lat, exe.end_lon, exe.end_at))
    return _DriverEval(True, costs, origins)


_EXE_CACHE: dict = {}


def _cached_execute(group: Group, lat: float, lon: float, free_at: datetime, driver: Driver, now: Optional[datetime]):
    # 用司机对象本身做键。电话相同但班次不同的两条记录不能共用一次模拟。
    key = (id(group), id(driver), lat, lon, free_at, now)
    hit = _EXE_CACHE.get(key)
    if hit is not None:
        return hit
    exe = execute_group(group, lat, lon, free_at, driver=driver, now=now, policy="jit", need_latest=False)
    _EXE_CACHE[key] = exe
    return exe


def _chain_cost(
    driver: Driver,
    groups: Sequence[Group],
    origin: Tuple[float, float, datetime],
    now: Optional[datetime],
    weights: LossWeights,
) -> Optional[float]:
    lat, lon, t = origin
    cost = 0.0
    for group in groups:
        exe = _cached_execute(group, lat, lon, t, driver, now)
        if not exe.feasible or exe.end_at is None:
            return None
        cost += execution_cost(exe, weights)
        lat, lon, t = exe.end_lat, exe.end_lon, exe.end_at
    return cost


def _solution_search_cost(evals: Sequence[_DriverEval], unassigned: Sequence[Group], weights: LossWeights) -> float:
    if any(not ev.feasible for ev in evals):
        return 1e12
    return sum(ev.cost for ev in evals) + _miss_penalty(unassigned, weights)


def search_cost(sol: Solution, now: Optional[datetime] = None, weights: LossWeights = DEFAULT_WEIGHTS) -> float:
    # 和 solution_cost 同一个 L，但不二分最晚出发。
    evals = [_eval_driver(plan, sol.locked_gids, now, weights) for plan in sol.plans]
    return _solution_search_cost(evals, sol.unassigned, weights)


def _candidate_places(
    sol: Solution,
    evals: Sequence[_DriverEval],
    group: Group,
    now: Optional[datetime],
    weights: LossWeights,
) -> List[Tuple[float, int, int]]:
    # 每个能接的司机留一个最好位置：(方案 L, 司机下标, 插入位置)
    if any(not ev.feasible for ev in evals):
        return []
    base = _solution_search_cost(evals, sol.unassigned, weights)
    locked = sol.locked_gids
    rows: List[Tuple[float, int, int]] = []
    for p_idx, plan in enumerate(sol.plans):
        if not plan.driver.consider_for(group):
            continue
        ev = evals[p_idx]
        best: Optional[Tuple[float, int, int]] = None
        lo = _min_pos(plan, locked)
        for pos in range(lo, len(plan.groups) + 1):
            prefix = sum(ev.costs[:pos])
            suffix = _chain_cost(
                plan.driver, (group, *plan.groups[pos:]), ev.origins[pos], now, weights
            )
            if suffix is None:
                continue
            total = base - ev.cost + prefix + suffix
            if best is None or total < best[0]:
                best = (total, p_idx, pos)
        if best is not None:
            rows.append(best)
    return rows


def _best_place(
    rows: Sequence[Tuple[float, int, int]],
) -> Optional[Tuple[float, int, int]]:
    best: Optional[Tuple[float, int, int]] = None
    for row in rows:
        if best is None or row[0] < best[0]:
            best = row
    return best


def _option_on_driver(
    sol: Solution,
    ev: _DriverEval,
    plan: DriverPlan,
    group: Group,
    now: Optional[datetime],
    weights: LossWeights,
) -> Optional[Tuple[float, int]]:
    # 这个司机把组插到最好位置后的整条链成本，以及插入下标。
    if not ev.feasible or not plan.driver.consider_for(group):
        return None
    best: Optional[Tuple[float, int]] = None
    lo = _min_pos(plan, sol.locked_gids)
    for pos in range(lo, len(plan.groups) + 1):
        prefix = sum(ev.costs[:pos])
        suffix = _chain_cost(plan.driver, (group, *plan.groups[pos:]), ev.origins[pos], now, weights)
        if suffix is None:
            continue
        local = prefix + suffix
        if best is None or local < best[0]:
            best = (local, pos)
    return best


def _full_cost(base: float, ev: _DriverEval, local: float) -> float:
    return base - ev.cost + local


def try_insert(
    sol: Solution,
    group: Group,
    now: Optional[datetime] = None,
    weights: LossWeights = DEFAULT_WEIGHTS,
) -> bool:
    evals = [_eval_driver(plan, sol.locked_gids, now, weights) for plan in sol.plans]
    place = _best_place(_candidate_places(sol, evals, group, now, weights))
    if place is None:
        sol.unassigned.append(group)
        return False
    _cost, p_idx, pos = place
    sol.plans[p_idx].groups.insert(pos, group)
    return True


def _seed_locked(drivers: Sequence[Driver], groups: Sequence[Group]) -> Tuple[Solution, List[Group]]:
    by_id = {g.gid: g for g in groups}
    sol = empty_solution(drivers)
    locked = []
    used = set()
    for plan in sol.plans:
        gid = plan.driver.current_gid
        if plan.driver.is_enroute() and gid and gid in by_id:
            plan.groups.append(by_id[gid])
            locked.append(gid)
            used.add(gid)
    sol.locked_gids = locked
    rest = [g for g in groups if g.gid not in used]
    return sol, rest


def _require_unique(drivers: Sequence[Driver], groups: Sequence[Group]) -> None:
    gids = [g.gid for g in groups]
    dup_gids = sorted({gid for gid in gids if gids.count(gid) > 1})
    if dup_gids:
        raise ValueError("配车单号重复：" + "、".join(dup_gids))
    dids = [d.did for d in drivers]
    dup_dids = sorted({did for did in dids if dids.count(did) > 1})
    if dup_dids:
        raise ValueError("司机身份重复：" + "、".join(dup_dids))
    owner: dict[str, str] = {}
    for driver in drivers:
        gid = driver.current_gid
        if not gid:
            continue
        if not driver.is_enroute():
            raise ValueError(
                f"司机 {driver.did} 是空闲状态，却带了在跑配车单 {gid}。"
                "在跑要标成 enroute，并给出本趟终点"
            )
        if gid in owner:
            raise ValueError(f"在跑的配车单 {gid} 同时挂在 {owner[gid]} 和 {driver.did}")
        owner[gid] = driver.did


def greedy_assign(drivers: Sequence[Driver], groups: Sequence[Group], now: Optional[datetime] = None) -> Solution:
    _require_unique(drivers, groups)
    _EXE_CACHE.clear()
    sol, rest = _seed_locked(drivers, groups)
    for group in sorted(rest, key=lambda g: (g.first.eta, g.gid)):
        try_insert(sol, group, now=now)
    return sol


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
        self.w_destroy = [1.0, 1.0, 1.0]
        self._scores = [0.0, 0.0, 0.0]
        self._uses = [0, 0, 0]
        self._rho = 0.2
        self._segment = 10
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
        sim = simulate_solution(sol, now=self.now, need_latest=False)
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

    def _repair(self, sol: Solution, groups: Sequence[Group]) -> float:
        # 未派也进池。优先插「只有一个人能接」的组，避免再按 ETA 占死人。
        # 每个组在各司机上的最好插法先算好；插进谁，下一轮只重算这个司机。
        weights = DEFAULT_WEIGHTS
        remaining = list(groups)
        evals = [_eval_driver(plan, sol.locked_gids, self.now, weights) for plan in sol.plans]
        options: dict[str, List[Optional[Tuple[float, int]]]] = {
            group.gid: [
                _option_on_driver(sol, ev, plan, group, self.now, weights)
                for plan, ev in zip(sol.plans, evals)
            ]
            for group in remaining
        }
        while remaining:
            base = _solution_search_cost(evals, sol.unassigned, weights)
            pick = remaining[0]
            pick_key = None
            pick_place: Optional[Tuple[float, int, int]] = None
            for group in remaining:
                costs: List[float] = []
                place: Optional[Tuple[float, int, int]] = None
                for p_idx, opt in enumerate(options[group.gid]):
                    if opt is None:
                        continue
                    full = _full_cost(base, evals[p_idx], opt[0])
                    costs.append(full)
                    if place is None or full < place[0]:
                        place = (full, p_idx, opt[1])
                costs.sort()
                if not costs:
                    regret = -1.0
                elif len(costs) == 1:
                    regret = 1e12
                else:
                    regret = costs[1] - costs[0]
                key = (regret, group.people(), group.first.eta, group.gid)
                if pick_key is None or key > pick_key:
                    pick, pick_key, pick_place = group, key, place
            remaining.remove(pick)
            options.pop(pick.gid, None)
            if pick_place is None:
                sol.unassigned.append(pick)
                continue
            _cost, p_idx, pos = pick_place
            sol.plans[p_idx].groups.insert(pos, pick)
            evals[p_idx] = _eval_driver(sol.plans[p_idx], sol.locked_gids, self.now, weights)
            plan = sol.plans[p_idx]
            for group in remaining:
                options[group.gid][p_idx] = _option_on_driver(
                    sol, evals[p_idx], plan, group, self.now, weights
                )
        return _solution_search_cost(evals, sol.unassigned, weights)

    def _note_destroy(self, op: int, reward: float) -> None:
        self._uses[op] += 1
        self._scores[op] += reward

    def _adapt_weights(self) -> None:
        # 每段按「新最优 / 优于当前 / 接受劣解」更新轮盘权重。
        for i, used in enumerate(self._uses):
            if not used:
                continue
            updated = (1.0 - self._rho) * self.w_destroy[i] + self._rho * (self._scores[i] / used)
            self.w_destroy[i] = max(updated, 0.05)
        self._scores = [0.0, 0.0, 0.0]
        self._uses = [0, 0, 0]

    def run(self, verbose: bool = True) -> Solution:
        _EXE_CACHE.clear()
        current = greedy_assign(self.drivers, self.groups, now=self.now)
        best = _clone(current)
        best_cost = search_cost(best, now=self.now)
        curr_cost = best_cost
        if verbose:
            print(
                f"初始贪心：已派 {len(best.assigned_groups())}/{len(self.groups)}，"
                f"L={best_cost:.1f}，未派 {len(best.unassigned)}",
                flush=True,
            )
        for it in range(self.max_iter):
            if verbose and it and it % 10 == 0:
                print(f"迭代{it}/{self.max_iter}  当前 L={curr_cost:.1f}  最优 L={best_cost:.1f}", flush=True)
            cand = _clone(current)
            op = self.roulette(self.w_destroy)
            removed = self.destroy_ops[op](cand)
            pool = removed + list(cand.unassigned)
            cand.unassigned = []
            new_cost = self._repair(cand, pool)
            reward = 0.0
            if new_cost < 1e12:
                accept = new_cost <= curr_cost
                if not accept:
                    delta = new_cost - curr_cost
                    accept = math.exp(-delta / max(self.temp, 1e-8)) > self.rng.random()
                if accept:
                    improved = new_cost + 1e-6 < curr_cost
                    current = cand
                    curr_cost = new_cost
                    if new_cost + 1e-6 < best_cost:
                        reward = 5.0
                        best = _clone(cand)
                        best_cost = new_cost
                        if verbose:
                            print(
                                f"迭代{it} 更优：已派 {len(best.assigned_groups())}/{len(self.groups)}，"
                                f"L={best_cost:.1f}，未派 {len(best.unassigned)}",
                                flush=True,
                            )
                    elif improved:
                        reward = 3.0
                    else:
                        reward = 1.0
            self._note_destroy(op, reward)
            if (it + 1) % self._segment == 0:
                self._adapt_weights()
            self.temp *= self.cool
        if verbose:
            print(
                f"ALNS {self.max_iter} 轮结束：最优已派 {len(best.assigned_groups())}/{len(self.groups)}，"
                f"L={best_cost:.1f}，未派 {len(best.unassigned)}",
                flush=True,
            )
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


def _clock(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    return value.strftime("%m-%d %H:%M")


def _guest_lines(group: Group) -> List[str]:
    action = "接客" if group.is_to_station() else "送达"
    lines = []
    for index, stop in enumerate(group.stops, start=1):
        who = stop.guest or stop.name or stop.oid
        lines.append(
            f"    {index}. {who}  {stop.people}人  预估{action} {_clock(stop.eta)}"
        )
        if stop.address:
            lines.append(f"       {stop.address}")
        lines.append(f"       {stop.lat:.6f}, {stop.lon:.6f}")
    return lines


def format_dispatch(sol: Solution, now: Optional[datetime] = None) -> str:
    sim = simulate_solution(sol, now=now, policy="jit")
    total = solution_loss(sol, now=now)
    n_songji = sum(1 for g in sol.assigned_groups() if g.is_to_station())
    n_jieji = sum(1 for g in sol.assigned_groups() if not g.is_to_station())
    loss_text = f"L={total.total:.0f}" if not total.infeasible else "L=∞"
    lines = [
        "======== 派单 ========",
        f"已派 {len(sol.assigned_groups())}（送机 {n_songji} / 接机 {n_jieji}）"
        f"  未派 {len(sol.unassigned)}  {loss_text}",
    ]
    if not sim.feasible:
        lines.append(f"方案不可行：{sim.reason}")
        return "\n".join(lines)

    for plan, ds in zip(sol.plans, sim.driver_sims):
        d = plan.driver
        if not plan.groups:
            lines.append(f"\n{d.name}  车号{d.plate}  待命")
            continue
        lines.append(f"\n{d.name}  车号{d.plate}  {len(plan.groups)} 个配车单")
        for group, exe in zip(plan.groups, ds.executions):
            if group.gid in sol.locked_gids:
                when = f"当前在跑，{_clock(d.free_at)} 到终点后接下单"
            else:
                when = (
                    f"建议派单 {_clock(exe.dispatch_at)}  "
                    f"出发 {_clock(exe.depart)}  "
                    f"最晚 {_clock(exe.latest_depart)}"
                )
            overtime = ""
            if exe.overtime and exe.end_at is not None and d.shift_end is not None:
                overtime = f"  加班到 {_clock(exe.end_at)}"
            lines.append(f"  {group.gid}  {group.kind_label()}  {when}{overtime}")
            lines.extend(_guest_lines(group))

    if sol.unassigned:
        lines.append("\n未派配车单：")
        for group in sol.unassigned:
            lines.append(f"  {group.gid}  {group.kind_label()}  首客预估 {_clock(group.first.eta)}")
            lines.extend(_guest_lines(group))
    return "\n".join(lines)
