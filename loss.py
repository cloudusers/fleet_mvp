# L(d,g) = θe·空驶 + θu·绕路 + θδ·晚点 + θw·空等 + θr·松弛不足 + θot·加班
# L(方案) = ΣL(d,g) + 未派罚。不可行记 1e12。组内载客公里不进 L。

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossWeights:
    empty_min: float = 1.2  # 空驶到开单点
    detour_km: float = 2.0  # 送机绕路；接机为 0
    eta_delay_min: float = 6.0
    hotel_wait_min: float = 3.0
    slack_risk_min: float = 0.8
    slack_safe_min: float = 8.0  # 距 late 少于此值开始计风险
    overtime_min: float = 2.0
    unassigned_group: float = 10000.0
    unassigned_person: float = 80.0


DEFAULT_WEIGHTS = LossWeights()


@dataclass
class LossBreakdown:
    empty: float = 0.0
    detour: float = 0.0
    delay: float = 0.0
    wait: float = 0.0
    slack_risk: float = 0.0
    overtime: float = 0.0
    miss: float = 0.0
    empty_min: float = 0.0
    empty_km: float = 0.0
    detour_km: float = 0.0
    eta_delay_min: float = 0.0
    hotel_wait_min: float = 0.0
    slack_min: float = 0.0
    risk_raw: float = 0.0
    overtime_min: float = 0.0
    unassigned_groups: int = 0
    unassigned_people: int = 0
    infeasible: bool = False

    @property
    def total(self) -> float:
        if self.infeasible:
            return 1e12
        return self.empty + self.detour + self.delay + self.wait + self.slack_risk + self.overtime + self.miss

    def explain(self, weights: LossWeights = DEFAULT_WEIGHTS) -> str:
        if self.infeasible:
            return "L=∞ 不可行"
        parts = [
            f"{weights.empty_min:g}×{self.empty_min:.1f}min空驶",
            f"{weights.detour_km:g}×{self.detour_km:.1f}km不顺路",
            f"{weights.eta_delay_min:g}×{self.eta_delay_min:.1f}晚",
            f"{weights.hotel_wait_min:g}×{self.hotel_wait_min:.1f}空等",
            f"{weights.slack_risk_min:g}×{self.risk_raw:.1f}风险",
        ]
        if self.overtime_min:
            parts.append(f"{weights.overtime_min:g}×{self.overtime_min:.1f}加班")
        if self.unassigned_groups:
            parts.append(f"{weights.unassigned_group:g}×{self.unassigned_groups}未派组")
            parts.append(f"{weights.unassigned_person:g}×{self.unassigned_people}未派人")
        return f"L={self.total:.1f} = " + " + ".join(parts)


def assignment_loss(exe, weights: LossWeights = DEFAULT_WEIGHTS) -> LossBreakdown:
    if exe is None or not exe.feasible:
        return LossBreakdown(infeasible=True)
    slacks = list(exe.stop_slacks) if exe.stop_slacks else [exe.slack_min]
    risk_raw = sum(max(0.0, weights.slack_safe_min - s) for s in slacks)
    ot = exe.overtime_min or 0.0
    return LossBreakdown(
        empty=weights.empty_min * exe.empty_min,
        detour=weights.detour_km * exe.detour_km,
        delay=weights.eta_delay_min * exe.eta_delay_min,
        wait=weights.hotel_wait_min * exe.hotel_wait_min,
        slack_risk=weights.slack_risk_min * risk_raw,
        overtime=weights.overtime_min * ot,
        empty_min=exe.empty_min,
        empty_km=exe.empty_km,
        detour_km=exe.detour_km,
        eta_delay_min=exe.eta_delay_min,
        hotel_wait_min=exe.hotel_wait_min,
        slack_min=exe.slack_min,
        risk_raw=risk_raw,
        overtime_min=ot,
    )


def solution_loss(
    sol, now=None, policy: str = "jit", weights: LossWeights = DEFAULT_WEIGHTS, need_latest: bool = True
) -> LossBreakdown:
    from fleet_mvp.execute import simulate_solution

    sim = simulate_solution(sol, now=now, policy=policy, need_latest=need_latest)
    if not sim.feasible:
        return LossBreakdown(infeasible=True)
    acc = LossBreakdown()
    for ds in sim.driver_sims:
        for exe in ds.executions:
            part = assignment_loss(exe, weights)
            acc.empty += part.empty
            acc.detour += part.detour
            acc.delay += part.delay
            acc.wait += part.wait
            acc.slack_risk += part.slack_risk
            acc.overtime += part.overtime
            acc.empty_min += part.empty_min
            acc.empty_km += part.empty_km
            acc.detour_km += part.detour_km
            acc.eta_delay_min += part.eta_delay_min
            acc.hotel_wait_min += part.hotel_wait_min
            acc.slack_min += part.slack_min
            acc.risk_raw += part.risk_raw
            acc.overtime_min += part.overtime_min
    people = sum(g.people() for g in sol.unassigned)
    acc.unassigned_groups = len(sol.unassigned)
    acc.unassigned_people = people
    acc.miss = weights.unassigned_group * acc.unassigned_groups + weights.unassigned_person * people
    return acc


def execution_cost(exe, weights: LossWeights = DEFAULT_WEIGHTS) -> float:
    return assignment_loss(exe, weights).total


def solution_cost(
    sol, now=None, policy: str = "jit", weights: LossWeights = DEFAULT_WEIGHTS, need_latest: bool = True
) -> float:
    return solution_loss(sol, now=now, policy=policy, weights=weights, need_latest=need_latest).total
