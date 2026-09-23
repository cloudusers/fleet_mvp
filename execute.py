# 按站序走完一趟。下一单的 origin 用本趟 end_lat/lon/end_at。

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple

from fleet_mvp.config import PREP_BUFFER_MIN, TRANSFER_STATION
from fleet_mvp.geo import add_minutes, detour_km, haversine_km, near_transfer, travel_minutes
from fleet_mvp.models import Driver, DriverPlan, DriverSim, Group, GroupExecution, Solution, SolutionSim, Stop


def _fail(reason: str, latest_depart=None) -> GroupExecution:
    return GroupExecution(feasible=False, reason=reason, latest_depart=latest_depart)


def _apply_shift_end(driver: Optional[Driver], exe: GroupExecution) -> GroupExecution:
    # 下班后才出发：不可行。班内出发、收车超时：记加班，仍可派。
    if driver is None or not exe.feasible:
        return exe
    if driver.shift_end is None:
        return exe
    if exe.depart is not None and exe.depart >= driver.shift_end:
        return _fail("已过下班时刻", exe.latest_depart)
    if exe.end_at is not None and exe.end_at > driver.shift_end:
        exe.overtime = True
        exe.overtime_min = (exe.end_at - driver.shift_end).total_seconds() / 60.0
        exe.reason = (
            f"加班：预计 {exe.end_at.strftime('%m-%d %H:%M')} 收车，"
            f"下班 {driver.shift_end.strftime('%m-%d %H:%M')}，超时 {exe.overtime_min:.0f} 分钟"
        )
    return exe


def _windows(arrive, stop: Stop) -> Tuple[float, float, float]:
    wait = max(0.0, (stop.eta - arrive).total_seconds() / 60.0)
    delay = max(0.0, (arrive - stop.eta).total_seconds() / 60.0)
    slack = (stop.late - arrive).total_seconds() / 60.0
    return wait, delay, slack


@dataclass
class _Walk:
    feasible: bool
    reason: str = ""
    wait_min: float = 0.0
    delay_min: float = 0.0
    slack_min: float = 0.0
    slacks: List[float] = field(default_factory=list)
    end_at: Optional[datetime] = None
    end_lat: float = 0.0
    end_lon: float = 0.0
    loaded_km: float = 0.0
    loaded_min: float = 0.0


def _walk_guest_stops(stops: Sequence[Stop], arrive_first: datetime, action: str) -> _Walk:
    # 站序固定。预估时刻已经含上下车，这里不再加停靠。任一站破 late，整组失败。
    if not stops:
        return _Walk(feasible=False, reason="空分组")
    wait_sum = delay_sum = 0.0
    slacks: List[float] = []
    t = arrive_first
    lat, lon = stops[0].lat, stops[0].lon
    start = arrive_first
    loaded_km = 0.0
    for i, stop in enumerate(stops):
        if i:
            loaded_km += haversine_km(lat, lon, stop.lat, stop.lon)
            t = add_minutes(t, travel_minutes(lat, lon, stop.lat, stop.lon))
        if t > stop.late:
            where = "首站" if i == 0 else "途经"
            return _Walk(feasible=False, reason=f"赶不上{where}{action} {stop.oid}")
        wait, delay, slack = _windows(t, stop)
        wait_sum += wait
        delay_sum += delay
        slacks.append(slack)
        t = max(t, stop.eta)
        lat, lon = stop.lat, stop.lon
    return _Walk(
        feasible=True,
        wait_min=wait_sum,
        delay_min=delay_sum,
        slack_min=min(slacks) if slacks else 0.0,
        slacks=slacks,
        end_at=t,
        end_lat=lat,
        end_lon=lon,
        loaded_km=loaded_km,
        loaded_min=(t - start).total_seconds() / 60.0,
    )


def _latest_ok(ok, lo: datetime, hi: datetime) -> Optional[datetime]:
    # 越晚越难。窗口里没有可行出发点时返回空，避免写出一个其实走不了的时刻。
    if hi < lo:
        hi = lo
    if not ok(lo):
        return None
    if ok(hi):
        return hi
    for _ in range(24):
        mid = lo + (hi - lo) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _dispatch_clock(depart, free_at, now):
    # 出发前 PREP_BUFFER 发单，不能早于 now / free_at
    dispatch_at = add_minutes(depart, -PREP_BUFFER_MIN)
    if now is not None and dispatch_at < now:
        dispatch_at = now
    if dispatch_at < free_at:
        dispatch_at = free_at
    if dispatch_at > depart:
        dispatch_at = depart
    return dispatch_at


def execute_to_station(
    group: Group,
    olat: float,
    olon: float,
    free_at: datetime,
    now=None,
    policy="jit",
    need_latest: bool = True,
) -> GroupExecution:
    # origin → 酒店… → T，收车在 T。need_latest 只服务最终输出，搜索时关掉。
    first = group.first
    empty_min = travel_minutes(olat, olon, first.lat, first.lon)
    empty_km = haversine_km(olat, olon, first.lat, first.lon)

    def _ok(depart: datetime) -> bool:
        return _walk_guest_stops(
            group.stops, add_minutes(depart, empty_min), "接客"
        ).feasible

    if need_latest:
        lo = min(free_at, add_minutes(first.eta, -empty_min)) - timedelta(hours=2)
        hi = add_minutes(first.late, -empty_min)
        latest_depart = _latest_ok(_ok, lo, hi)
    else:
        latest_depart = None
    ideal_depart = add_minutes(first.eta, -empty_min)
    depart = free_at if policy == "immediate" else max(free_at, ideal_depart)
    # 决策时刻已经过了建议出发，不能按过去的时刻发车。
    if now is not None and depart < now:
        depart = now
    if need_latest and not _ok(depart):
        return _fail("出发后仍赶不上本组全程接客", latest_depart)

    arrive = add_minutes(depart, empty_min)
    walked = _walk_guest_stops(group.stops, arrive, "接客")
    if not walked.feasible:
        return _fail(walked.reason, latest_depart)

    tlat, tlon = TRANSFER_STATION
    to_t = travel_minutes(walked.end_lat, walked.end_lon, tlat, tlon)
    end_at = add_minutes(walked.end_at, to_t)
    extra = 0.0 if near_transfer(olat, olon) else detour_km(olat, olon, first.lat, first.lon)
    loaded_min = walked.loaded_min + to_t
    return GroupExecution(
        feasible=True,
        depart=depart,
        dispatch_at=_dispatch_clock(depart, free_at, now),
        latest_depart=latest_depart,
        arrive_first=arrive,
        hotel_wait_min=walked.wait_min,
        eta_delay_min=walked.delay_min,
        slack_min=walked.slack_min,
        stop_slacks=walked.slacks,
        loaded_min=loaded_min,
        empty_min=empty_min,
        empty_km=empty_km,
        detour_km=extra,
        trip_km=empty_km + walked.loaded_km + haversine_km(walked.end_lat, walked.end_lon, tlat, tlon),
        end_lat=tlat,
        end_lon=tlon,
        end_at=end_at,
    )


def execute_from_station(
    group: Group,
    olat: float,
    olon: float,
    free_at: datetime,
    now=None,
    policy="jit",
    need_latest: bool = True,
) -> GroupExecution:
    # origin → T → 酒店…，收车在末户。need_latest 只服务最终输出，搜索时关掉。
    tlat, tlon = TRANSFER_STATION
    empty_min = travel_minutes(olat, olon, tlat, tlon)
    empty_km = haversine_km(olat, olon, tlat, tlon)
    first = group.first
    to_first = travel_minutes(tlat, tlon, first.lat, first.lon)
    ideal_leave_t = add_minutes(first.eta, -to_first)

    if need_latest:
        def _from_origin(origin_depart: datetime) -> _Walk:
            arrive_t = add_minutes(origin_depart, empty_min)
            return _walk_guest_stops(
                group.stops, add_minutes(arrive_t, to_first), "送达"
            )

        lo = free_at - timedelta(hours=2)
        hi = add_minutes(first.late, -to_first - empty_min)
        latest_depart = _latest_ok(lambda d: _from_origin(d).feasible, lo, hi)
    else:
        latest_depart = None

    if policy == "immediate":
        depart = free_at
    else:
        need_leave_origin = add_minutes(ideal_leave_t, -empty_min)
        depart = max(free_at, need_leave_origin)
    if now is not None and depart < now:
        depart = now
    arrive_t = add_minutes(depart, empty_min)
    leave_ready = arrive_t if policy == "immediate" else max(arrive_t, ideal_leave_t)

    arrive = add_minutes(leave_ready, to_first)
    walked = _walk_guest_stops(group.stops, arrive, "送达")
    if not walked.feasible:
        return _fail(walked.reason, latest_depart)

    return GroupExecution(
        feasible=True,
        depart=depart,
        dispatch_at=_dispatch_clock(depart, free_at, now),
        latest_depart=latest_depart,
        arrive_first=arrive,
        hotel_wait_min=walked.wait_min,
        eta_delay_min=walked.delay_min,
        slack_min=walked.slack_min,
        stop_slacks=walked.slacks,
        loaded_min=walked.loaded_min,
        empty_min=empty_min,
        empty_km=empty_km,
        detour_km=0.0,
        trip_km=empty_km + haversine_km(tlat, tlon, first.lat, first.lon) + walked.loaded_km,
        end_lat=walked.end_lat,
        end_lon=walked.end_lon,
        end_at=walked.end_at,
    )


def execute_group(
    group: Group,
    origin_lat: float,
    origin_lon: float,
    free_at: datetime,
    driver: Optional[Driver] = None,
    now: Optional[datetime] = None,
    policy: str = "jit",
    need_latest: bool = True,
) -> GroupExecution:
    if not group.stops:
        return _fail("空分组")
    if driver is not None:
        if not driver.on_shift:
            return _fail("下班，不参与派单")
    if group.is_to_station():
        exe = execute_to_station(
            group, origin_lat, origin_lon, free_at, now=now, policy=policy, need_latest=need_latest
        )
    else:
        exe = execute_from_station(
            group, origin_lat, origin_lon, free_at, now=now, policy=policy, need_latest=need_latest
        )
    return _apply_shift_end(driver, exe)


def execute_for_driver(
    group: Group, driver: Driver, now=None, policy: str = "jit", need_latest: bool = True
) -> GroupExecution:
    lat, lon, t = driver.dispatch_origin()
    return execute_group(group, lat, lon, t, driver=driver, now=now, policy=policy, need_latest=need_latest)


def _locked_in_progress(driver: Driver) -> GroupExecution:
    # 在跑的单已经写进 dest/free_at，别从 GPS 再算一遍空驶
    lat, lon, t = driver.dispatch_origin()
    return GroupExecution(
        feasible=True,
        reason="in_progress",
        depart=t,
        dispatch_at=t,
        latest_depart=t,
        arrive_first=t,
        slack_min=8.0,
        end_lat=lat,
        end_lon=lon,
        end_at=t,
    )


def simulate_driver_plan(
    plan: DriverPlan,
    now: Optional[datetime] = None,
    policy: str = "jit",
    locked_gids: Optional[Sequence[str]] = None,
    need_latest: bool = True,
) -> DriverSim:
    lat, lon, t = plan.driver.dispatch_origin()
    locked = set(locked_gids or [])
    execs: List[GroupExecution] = []
    for i, group in enumerate(plan.groups):
        if i == 0 and group.gid in locked:
            exe = _locked_in_progress(plan.driver)
        else:
            exe = execute_group(
                group, lat, lon, t, driver=plan.driver, now=now, policy=policy, need_latest=need_latest
            )
        if not exe.feasible:
            return DriverSim(did=plan.driver.did, feasible=False, executions=execs, reason=exe.reason)
        execs.append(exe)
        lat, lon = exe.end_lat, exe.end_lon
        t = exe.end_at
    return DriverSim(
        did=plan.driver.did,
        feasible=True,
        executions=execs,
        km=sum(e.trip_km for e in execs),
        empty_km=sum(e.empty_km for e in execs),
        hotel_wait_min=sum(e.hotel_wait_min for e in execs),
        eta_delay_min=sum(e.eta_delay_min for e in execs),
    )


def simulate_solution(
    sol: Solution, now: Optional[datetime] = None, policy: str = "jit", need_latest: bool = True
) -> SolutionSim:
    sims: List[DriverSim] = []
    for plan in sol.plans:
        ds = simulate_driver_plan(
            plan, now=now, policy=policy, locked_gids=sol.locked_gids, need_latest=need_latest
        )
        if not ds.feasible:
            return SolutionSim(
                feasible=False,
                driver_sims=sims + [ds],
                unassigned=list(sol.unassigned),
                reason=f"{plan.driver.did}: {ds.reason}",
            )
        sims.append(ds)
    return SolutionSim(feasible=True, driver_sims=sims, unassigned=list(sol.unassigned))
