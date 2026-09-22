# haversine + 均速，没接路网

from __future__ import annotations

import math
from datetime import datetime, timedelta

from fleet_mvp.config import AVERAGE_SPEED_KMH, AT_TRANSFER_KM, TRANSFER_STATION


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def travel_minutes(lat1: float, lon1: float, lat2: float, lon2: float, speed_kmh: float = AVERAGE_SPEED_KMH) -> float:
    if speed_kmh <= 0:
        raise ValueError("speed_kmh must be positive")
    km = haversine_km(lat1, lon1, lat2, lon2)
    return km / speed_kmh * 60.0


def add_minutes(t: datetime, minutes: float) -> datetime:
    return t + timedelta(minutes=minutes)


def detour_km(origin_lat: float, origin_lon: float, first_lat: float, first_lon: float) -> float:
    # origin→首站→T 相对 origin→T 多走的公里
    tlat, tlon = TRANSFER_STATION
    via = haversine_km(origin_lat, origin_lon, first_lat, first_lon) + haversine_km(
        first_lat, first_lon, tlat, tlon
    )
    direct = haversine_km(origin_lat, origin_lon, tlat, tlon)
    return max(0.0, via - direct)


def near_transfer(lat: float, lon: float, thresh_km: float = AT_TRANSFER_KM) -> bool:
    tlat, tlon = TRANSFER_STATION
    return haversine_km(lat, lon, tlat, tlon) <= thresh_km
