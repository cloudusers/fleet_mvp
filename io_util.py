# 读 data/*.json

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

from fleet_mvp.config import DRIVER_ENROUTE, DRIVER_IDLE, KIND_FROM_STATION, KIND_TO_STATION, TRANSFER_STATION
from fleet_mvp.models import Driver, Group, Stop


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_drivers(path: Union[str, Path]) -> List[Driver]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    drivers = []
    for row in raw:
        status = str(row.get("status", DRIVER_IDLE))
        dest_lat = row.get("dest_lat")
        dest_lon = row.get("dest_lon")
        if dest_lat is not None:
            dest_lat = float(dest_lat)
            dest_lon = float(dest_lon)
        elif status == DRIVER_ENROUTE:
            dest_lat, dest_lon = TRANSFER_STATION
        drivers.append(
            Driver(
                did=str(row["did"]),
                name=str(row.get("name", row["did"])),
                plate=str(row.get("plate", "")),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                free_at=_parse_dt(row["free_at"]),
                status=status,
                current_kind=row.get("current_kind"),
                current_gid=row.get("current_gid"),
                dest_lat=dest_lat,
                dest_lon=dest_lon,
                note=str(row.get("note", "")),
                on_shift=bool(row.get("on_shift", True)),
                shift_start=_parse_dt(row["shift_start"]) if row.get("shift_start") else None,
                shift_end=_parse_dt(row["shift_end"]) if row.get("shift_end") else None,
            )
        )
    return drivers


def load_groups(path: Union[str, Path]) -> List[Group]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = []
    for row in raw:
        kind = str(row["kind"])
        if kind not in (KIND_TO_STATION, KIND_FROM_STATION):
            raise ValueError(f"未知 kind={kind}，应为 to_station 或 from_station")
        stops = []
        for s in row["stops"]:
            stops.append(
                Stop(
                    oid=str(s["oid"]),
                    name=str(s["name"]),
                    lat=float(s["lat"]),
                    lon=float(s["lon"]),
                    eta=_parse_dt(s["eta"]),
                    late=_parse_dt(s["late"]),
                    people=int(s["people"]),
                    guest=str(s.get("guest", "")),
                    address=str(s.get("address", "")),
                )
            )
        if not stops:
            raise ValueError(f"配车单 {row.get('gid')} 没有客人，不能丢掉")
        groups.append(Group(gid=str(row["gid"]), kind=kind, stops=stops, note=str(row.get("note", ""))))
    return groups


def data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def default_dataset(
    peak: bool = False, trap: bool = False, real: bool = False
) -> Tuple[List[Driver], List[Group]]:
    picked = [name for name, flag in (("peak", peak), ("trap", trap), ("real", real)) if flag]
    if len(picked) > 1:
        raise ValueError("peak、trap、real 只能选一个")
    if real:
        return (
            load_drivers(data_dir() / "drivers_0922.json"),
            load_groups(data_dir() / "groups_0922.json"),
        )
    if trap:
        return (
            load_drivers(data_dir() / "drivers_trap.json"),
            load_groups(data_dir() / "groups_trap.json"),
        )
    drivers = load_drivers(data_dir() / "drivers.json")
    groups_name = "groups_peak.json" if peak else "groups.json"
    return drivers, load_groups(data_dir() / groups_name)
