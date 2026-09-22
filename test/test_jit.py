from __future__ import annotations

import unittest
from datetime import datetime

from fleet_mvp.config import PREP_BUFFER_MIN, TRANSFER_STATION
from fleet_mvp.execute import execute_group
from fleet_mvp.geo import haversine_km, travel_minutes
from fleet_mvp.models import Driver, Group, Stop


def _songji(**kwargs) -> Group:
    stop = dict(
        oid="S1",
        name="难波",
        lat=34.6639,
        lon=135.5019,
        eta=datetime(2026, 9, 20, 14, 20),
        late=datetime(2026, 9, 20, 14, 40),
        people=2,
    )
    stop.update(kwargs)
    return Group(gid="G", kind="to_station", stops=[Stop(**stop)])


class JitDispatchTests(unittest.TestCase):
    def test_do_not_leave_early_and_wait_at_hotel(self):
        driver = Driver("D2", 34.6639, 135.5019, datetime(2026, 9, 20, 14, 0))
        group = _songji()
        exe = execute_group(group, driver.lat, driver.lon, driver.free_at, driver=driver)
        self.assertTrue(exe.feasible)
        self.assertEqual(exe.hotel_wait_min, 0)
        self.assertEqual(exe.arrive_first, group.first.eta)
        self.assertLess(exe.dispatch_at, exe.depart)
        self.assertGreaterEqual(exe.depart, datetime(2026, 9, 20, 14, 10))

    def test_dispatch_is_before_depart_by_buffer(self):
        driver = Driver("D2", 34.6639, 135.5019, datetime(2026, 9, 20, 14, 0))
        group = _songji()
        exe = execute_group(group, driver.lat, driver.lon, driver.free_at, driver=driver)
        delta = (exe.depart - exe.dispatch_at).total_seconds() / 60.0
        self.assertAlmostEqual(delta, PREP_BUFFER_MIN, delta=0.1)

    def test_immediate_policy_creates_hotel_wait(self):
        driver = Driver("D2", 34.6639, 135.5019, datetime(2026, 9, 20, 14, 0))
        group = _songji()
        jit = execute_group(group, driver.lat, driver.lon, driver.free_at, driver=driver, policy="jit")
        now = execute_group(
            group, driver.lat, driver.lon, driver.free_at, driver=driver, policy="immediate"
        )
        self.assertEqual(jit.hotel_wait_min, 0)
        self.assertGreater(now.hotel_wait_min, 10)

    def test_far_driver_misses_tight_window(self):
        driver = Driver("D4", 34.6654, 135.4323, datetime(2026, 9, 20, 14, 0), note="USJ")
        group = _songji(
            lat=34.6212,
            lon=135.5510,
            eta=datetime(2026, 9, 20, 14, 0),
            late=datetime(2026, 9, 20, 14, 6),
        )
        exe = execute_group(group, driver.lat, driver.lon, driver.free_at, driver=driver)
        self.assertFalse(exe.feasible)

    def test_latest_depart_matches_travel_time(self):
        driver = Driver("D1", 34.7024, 135.4959, datetime(2026, 9, 20, 14, 0))
        group = _songji()
        drive = travel_minutes(driver.lat, driver.lon, group.first.lat, group.first.lon)
        exe = execute_group(group, driver.lat, driver.lon, driver.free_at, driver=driver)
        self.assertTrue(exe.feasible)
        expect = group.first.late.timestamp() - drive * 60
        self.assertAlmostEqual(exe.latest_depart.timestamp(), expect, delta=1)

    def test_near_driver_beats_far_on_empty(self):
        now = datetime(2026, 9, 20, 14, 0)
        group = _songji()
        near = Driver("D2", 34.6639, 135.5019, now)
        far = Driver("D4", 34.6654, 135.4323, now)
        e_near = execute_group(group, near.lat, near.lon, near.free_at, driver=near)
        e_far = execute_group(group, far.lat, far.lon, far.free_at, driver=far)
        self.assertTrue(e_near.feasible)
        self.assertLess(e_near.empty_km, e_far.empty_km)
        self.assertLess(e_near.detour_km, e_far.detour_km)

    def test_jieji_jit_waits_at_origin_not_at_station_early(self):
        group = Group(
            gid="J",
            kind="from_station",
            stops=[
                Stop(
                    oid="S1",
                    name="难波",
                    lat=34.6639,
                    lon=135.5019,
                    eta=datetime(2026, 9, 20, 15, 30),
                    late=datetime(2026, 9, 20, 16, 0),
                    people=2,
                )
            ],
        )
        hotel = Driver("D5", 34.6922, 135.4721, datetime(2026, 9, 20, 14, 0))
        jit = execute_group(group, hotel.lat, hotel.lon, hotel.free_at, driver=hotel, policy="jit")
        immediate = execute_group(
            group, hotel.lat, hotel.lon, hotel.free_at, driver=hotel, policy="immediate"
        )
        self.assertTrue(jit.feasible)
        self.assertGreater(jit.depart, hotel.free_at)
        self.assertGreater(immediate.hotel_wait_min, jit.hotel_wait_min)
        tlat, tlon = TRANSFER_STATION
        self.assertGreater(jit.empty_km, 0.5)
        self.assertAlmostEqual(
            jit.empty_km,
            haversine_km(hotel.lat, hotel.lon, tlat, tlon),
            places=2,
        )


if __name__ == "__main__":
    unittest.main()
