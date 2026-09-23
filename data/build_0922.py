# 把 2026-09-22 线上配车 / 订单表收成 drivers_0922.json、groups_0922.json。
# 一组 = 一个配车单。站序按预估到达客人点的时间，同一时刻保持表上的先后。

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from openpyxl import load_workbook

DATA_DIR = Path(__file__).resolve().parent
PKG = DATA_DIR.parent
REPO = PKG.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fleet_mvp.config import LATE_SLACK_MIN, TRANSFER_STATION

XLSX_DIR = PKG / "test"

SHIFT_START = datetime(2026, 9, 20, 20, 0, 0)
SHIFT_END = datetime(2026, 9, 23, 2, 0, 0)

ETA_RE = re.compile(
    r"(?P<eta>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"(?:\[(?P<flight>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\])?\s*$"
)
PAX_RE = re.compile(r"\((?P<adults>\d+),(?P<children>\d+)\+")
DRIVER_RE = re.compile(r"^(?P<name>.+?)(?:【(?P<plate>\d+)】)?$")


def _rows(path: Path, header_row: int) -> list[dict]:
    ws = load_workbook(path, data_only=True, read_only=True).active
    table = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c is not None else "" for c in table[header_row]]
    out = []
    for raw in table[header_row + 1 :]:
        if not any(c is not None and str(c).strip() for c in raw):
            continue
        out.append({header[i]: raw[i] if i < len(raw) else None for i in range(len(header))})
    return out


def _ffill(rows: list[dict], columns: list[str]) -> None:
    last = {col: None for col in columns}
    for row in rows:
        for col in columns:
            value = row.get(col)
            if value is None or str(value).strip() == "":
                row[col] = last[col]
            else:
                last[col] = value


def _text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_eta(passenger: str) -> datetime:
    matched = ETA_RE.search(passenger)
    if not matched:
        raise ValueError(f"旅客信息里没有预估时间：{passenger}")
    return datetime.strptime(matched.group("eta"), "%Y-%m-%d %H:%M:%S")


def _parse_driver(value: str) -> tuple[str, str, str]:
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    head = lines[0]
    phone = lines[1] if len(lines) > 1 else ""
    matched = DRIVER_RE.match(head)
    if not matched:
        raise ValueError(f"司机信息无法解析：{value!r}")
    name = matched.group("name").strip()
    plate = matched.group("plate") or ""
    if not phone:
        phone = name
    return name, plate, phone


def _parse_location(value: str) -> tuple[float, float]:
    lon_text, lat_text = _text(value).split(",")
    lat, lon = float(lat_text), float(lon_text)
    if not (34.0 <= lat <= 35.5 and 135.0 <= lon <= 136.2):
        raise ValueError(f"经纬度不在大阪附近：{value}")
    return lat, lon


def _orders(path: Path) -> dict[str, dict]:
    found = {}
    for row in _rows(path, header_row=0):
        order_no = _text(row.get("order_no"))
        if not order_no:
            continue
        found[order_no] = row
    return found


def _guest_name(passenger: str) -> str:
    cut = passenger.find("(")
    name = passenger[:cut].strip() if cut > 0 else passenger.strip()
    return name or passenger.strip()


def _people(order: dict, passenger: str) -> int:
    adults = int(order.get("person_num") or 0)
    children = int(order.get("child_num") or 0)
    if adults + children <= 0:
        matched = PAX_RE.search(passenger)
        if not matched:
            return 1
        adults = int(matched.group("adults"))
        children = int(matched.group("children"))
    return adults + children


def _load_dispatch(path: Path, orders: dict[str, dict], kind: str) -> tuple[list[dict], dict]:
    rows = _rows(path, header_row=1)
    _ffill(rows, ["配车单号", "司机信息", "送迎类型"])
    grouped: dict[str, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        order_no = _text(row.get("订单号"))
        gid = _text(row.get("配车单号"))
        if not order_no or not gid:
            raise ValueError(f"{path.name} 有行缺少配车单号或订单号")
        if order_no not in orders:
            raise ValueError(f"{order_no} 在订单表里找不到")
        order = orders[order_no]
        if _text(order.get("allocation_no")) != gid:
            raise ValueError(f"{order_no} 的配车单号对不上：配车表 {gid}，订单表 {order.get('allocation_no')}")
        passenger = _text(row.get("旅客姓名"))
        people = _people(order, passenger)
        lat, lon = _parse_location(order.get("departure_location"))
        eta = _parse_eta(passenger)
        name, plate, phone = _parse_driver(row.get("司机信息"))
        grouped[gid].append(
            {
                "index": index,
                "oid": order_no,
                "name": _text(order.get("departure_name")) or _text(order.get("departure_address")),
                "lat": lat,
                "lon": lon,
                "eta": eta,
                "people": people,
                "guest": _guest_name(passenger),
                "address": _text(order.get("departure_address")),
                "driver_name": name,
                "plate": plate,
                "phone": phone,
                "kind_label": _text(row.get("送迎类型")),
            }
        )
    groups = []
    history = {}
    for gid, stops in grouped.items():
        labels = {stop["kind_label"] for stop in stops}
        if labels != {("送机" if kind == "to_station" else "接机")}:
            raise ValueError(f"{gid} 送迎类型异常：{labels}")
        stops.sort(key=lambda stop: (stop["eta"], stop["index"]))
        driver_name = stops[0]["driver_name"]
        plate = stops[0]["plate"]
        phone = stops[0]["phone"]
        history[gid] = {"did": phone, "name": driver_name, "plate": plate}
        plate_text = f"【{plate}】" if plate else ""
        groups.append(
            {
                "gid": gid,
                "kind": kind,
                "note": f"历史司机{driver_name}{plate_text}",
                "stops": [
                    {
                        "oid": stop["oid"],
                        "name": stop["name"],
                        "lat": stop["lat"],
                        "lon": stop["lon"],
                        "eta": stop["eta"].isoformat(timespec="seconds"),
                        "late": (stop["eta"] + timedelta(minutes=LATE_SLACK_MIN)).isoformat(timespec="seconds"),
                        "people": stop["people"],
                        "guest": stop["guest"],
                        "address": stop["address"],
                    }
                    for stop in stops
                ],
            }
        )
    return groups, history


def _drivers(history: dict) -> list[dict]:
    bucket: dict[str, dict] = {}
    for item in history.values():
        rec = bucket.setdefault(
            item["did"],
            {"name": item["name"], "plates": defaultdict(int)},
        )
        if item["name"] != rec["name"]:
            raise ValueError(f"同一电话对应了两个姓名：{item['did']} {rec['name']} / {item['name']}")
        if item["plate"]:
            rec["plates"][item["plate"]] += 1
    drivers = []
    for phone, rec in bucket.items():
        plates = sorted(rec["plates"], key=lambda plate: (-rec["plates"][plate], plate))
        if plates:
            primary = plates[0]
            car_note = f"车号{primary}"
            if len(plates) > 1:
                car_note += "，同日也开过" + "、".join(plates[1:])
        else:
            primary = "无车号"
            car_note = "表里没有车号"
        drivers.append(
            {
                "did": phone,
                "name": rec["name"],
                "plate": primary,
                "status": "idle",
                "current_kind": None,
                "current_gid": None,
                "lat": TRANSFER_STATION[0],
                "lon": TRANSFER_STATION[1],
                "free_at": SHIFT_START.isoformat(timespec="seconds"),
                "note": f"9月22日实班，{car_note}。表里没有出车GPS，从中转站待命",
                "on_shift": True,
                "shift_start": SHIFT_START.isoformat(timespec="seconds"),
                "shift_end": SHIFT_END.isoformat(timespec="seconds"),
            }
        )
    drivers.sort(key=lambda row: row["name"])
    return drivers


def build() -> tuple[list[dict], list[dict]]:
    songji, songji_hist = _load_dispatch(
        XLSX_DIR / "dropoff-2026-09-22.xlsx",
        _orders(XLSX_DIR / "s0922.xlsx"),
        "to_station",
    )
    jieji, jieji_hist = _load_dispatch(
        XLSX_DIR / "pickup-2026-09-22.xlsx",
        _orders(XLSX_DIR / "j0922.xlsx"),
        "from_station",
    )
    groups = songji + jieji
    groups.sort(key=lambda group: (group["stops"][0]["eta"], group["gid"]))
    history = {**songji_hist, **jieji_hist}
    return _drivers(history), groups


def main() -> None:
    drivers, groups = build()
    DATA_DIR.mkdir(exist_ok=True)
    for name, payload in (
        ("drivers_0922.json", drivers),
        ("groups_0922.json", groups),
    ):
        path = DATA_DIR / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"写入 {path.name}")
    n_songji = sum(1 for group in groups if group["kind"] == "to_station")
    n_stops = sum(len(group["stops"]) for group in groups)
    off_day = 0
    for group in groups:
        for stop in group["stops"]:
            if not stop["eta"].startswith("2026-09-22"):
                off_day += 1
    print(
        f"司机 {len(drivers)}  编组 {len(groups)}（送机 {n_songji} / 接机 {len(groups) - n_songji}）"
        f"  客人 {n_stops}  预估时间不在 9月22日的站 {off_day}"
    )


if __name__ == "__main__":
    main()
