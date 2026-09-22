from __future__ import annotations

import unittest

from fleet_mvp.geo import detour_km, haversine_km, travel_minutes
from fleet_mvp.config import TRANSFER_STATION, AVERAGE_SPEED_KMH


class GeoTests(unittest.TestCase):
    def test_same_point_is_zero(self):
        lat, lon = TRANSFER_STATION
        self.assertAlmostEqual(haversine_km(lat, lon, lat, lon), 0.0, places=6)

    def test_osaka_station_to_namba_is_city_scale(self):
        km = haversine_km(34.7024, 135.4959, 34.6639, 135.5019)
        self.assertGreater(km, 3.0)
        self.assertLess(km, 8.0)

    def test_travel_minutes_match_speed(self):
        km = 14.0
        minutes = km / AVERAGE_SPEED_KMH * 60.0
        self.assertAlmostEqual(
            travel_minutes(34.66, 135.48, 34.66 + km / 111.0, 135.48),
            minutes,
            delta=1.5,
        )

    def test_detour_zero_when_first_stop_is_on_straight_line(self):
        extra = detour_km(*TRANSFER_STATION, *TRANSFER_STATION)
        self.assertAlmostEqual(extra, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
