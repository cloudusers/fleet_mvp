from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from fleet_mvp.config import (
    DRIVER_ENROUTE,
    DRIVER_IDLE,
    KIND_FROM_STATION,
    KIND_TO_STATION,
    TRANSFER_STATION,
)


@dataclass
class Stop:
    # eta/late：送机是接客窗，接机是送达窗
    oid: str
    name: str
    lat: float
    lon: float
    eta: datetime
    late: datetime
    people: int


@dataclass
class Group:
    # stops 只有客人点，T 不入库，方向看 kind
    gid: str
    kind: str = KIND_TO_STATION
    stops: List[Stop] = field(default_factory=list)
    note: str = ""

    @property
    def first(self) -> Stop:
        return self.stops[0]

    @property
    def last(self) -> Stop:
        return self.stops[-1]

    def is_to_station(self) -> bool:
        return self.kind == KIND_TO_STATION

    def kind_label(self) -> str:
        return "送机" if self.is_to_station() else "接机"

    def people(self) -> int:
        return sum(s.people for s in self.stops)

    def route_text(self) -> str:
        names = "-".join((s.name or s.oid) for s in self.stops)
        return f"{names}-T" if self.is_to_station() else f"T-{names}"


@dataclass
class Driver:
    did: str
    lat: float
    lon: float
    free_at: datetime
    name: str = ""
    plate: str = ""
    status: str = DRIVER_IDLE
    current_kind: Optional[str] = None
    current_gid: Optional[str] = None
    dest_lat: Optional[float] = None
    dest_lon: Optional[float] = None
    note: str = ""
    on_shift: bool = True
    shift_start: Optional[datetime] = None
    shift_end: Optional[datetime] = None

    def is_enroute(self) -> bool:
        return self.status == DRIVER_ENROUTE

    def current_kind_label(self) -> str:
        if self.current_kind == KIND_TO_STATION:
            return "送机"
        if self.current_kind == KIND_FROM_STATION:
            return "接机"
        return "待命"

    def dispatch_origin(self) -> Tuple[float, float, datetime]:
        # 空闲从 GPS 走；在跑的从本趟终点走。没到上班点就等到上班。
        if self.is_enroute():
            if self.dest_lat is None or self.dest_lon is None:
                tlat, tlon = TRANSFER_STATION
            else:
                tlat, tlon = self.dest_lat, self.dest_lon
            t = self.free_at
        else:
            tlat, tlon, t = self.lat, self.lon, self.free_at
        if self.shift_start is not None and t < self.shift_start:
            t = self.shift_start
        return tlat, tlon, t

    def consider_for(self, group: "Group") -> bool:
        # 下班 / 已过下班 / 首客 ETA 过了下班 / 赶不上 late → 不进候选
        if not self.on_shift:
            return False
        _lat, _lon, ready = self.dispatch_origin()
        if self.shift_end is not None:
            if ready >= self.shift_end:
                return False
            if group.first.eta >= self.shift_end:
                return False
        if ready > group.first.late:
            return False
        return True

    def status_label(self) -> str:
        if not self.on_shift:
            return "下班，不参与派单"
        shift = ""
        if self.shift_start and self.shift_end:
            shift = f" 班次{self.shift_start.strftime('%H:%M')}-{self.shift_end.strftime('%H:%M')}"
        if self.is_enroute():
            gid = self.current_gid or "?"
            return (
                f"运送中{self.current_kind_label()} {gid}，"
                f"{self.free_at.strftime('%H:%M')} 到达本趟终点后可接下单{shift}"
            )
        return f"空闲 {self.free_at.strftime('%H:%M')}{shift}"


@dataclass
class DriverPlan:
    driver: Driver
    groups: List[Group] = field(default_factory=list)


@dataclass
class Solution:
    plans: List[DriverPlan]
    unassigned: List[Group] = field(default_factory=list)
    locked_gids: List[str] = field(default_factory=list)

    def assigned_groups(self) -> List[Group]:
        out: List[Group] = []
        for plan in self.plans:
            out.extend(plan.groups)
        return out

    def all_gids(self) -> List[str]:
        return [g.gid for g in self.assigned_groups()] + [g.gid for g in self.unassigned]


@dataclass
class GroupExecution:
    feasible: bool
    reason: str = ""
    depart: Optional[datetime] = None
    dispatch_at: Optional[datetime] = None
    latest_depart: Optional[datetime] = None
    arrive_first: Optional[datetime] = None
    hotel_wait_min: float = 0.0
    eta_delay_min: float = 0.0
    slack_min: float = 0.0
    stop_slacks: List[float] = field(default_factory=list)
    loaded_min: float = 0.0
    empty_min: float = 0.0
    empty_km: float = 0.0
    detour_km: float = 0.0
    trip_km: float = 0.0
    end_lat: float = 0.0
    end_lon: float = 0.0
    end_at: Optional[datetime] = None
    overtime: bool = False
    overtime_min: float = 0.0


@dataclass
class DriverSim:
    did: str
    feasible: bool
    executions: List[GroupExecution] = field(default_factory=list)
    km: float = 0.0
    empty_km: float = 0.0
    hotel_wait_min: float = 0.0
    eta_delay_min: float = 0.0
    reason: str = ""


@dataclass
class SolutionSim:
    feasible: bool
    driver_sims: List[DriverSim]
    unassigned: List[Group]
    reason: str = ""
