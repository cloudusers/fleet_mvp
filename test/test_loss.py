from __future__ import annotations

import unittest
from datetime import datetime

from fleet_mvp.config import KIND_FROM_STATION, TRANSFER_STATION
from fleet_mvp.execute import execute_group
from fleet_mvp.loss import DEFAULT_WEIGHTS, assignment_loss, solution_loss
from fleet_mvp.models import Driver, DriverPlan, Group, Solution, Stop


def _namba_songji() -> Group:
    return Group(
        gid="G",
        kind="to_station",
        stops=[
            Stop(
                oid="S1",
                name="难波",
                lat=34.6639,
                lon=135.5019,
                eta=datetime(2026, 9, 20, 14, 20),
                late=datetime(2026, 9, 20, 14, 50),
                people=2,
            )
        ],
    )


def _namba_jieji() -> Group:
    return Group(
        gid="J",
        kind=KIND_FROM_STATION,
        stops=[
            Stop(
                oid="S1",
                name="难波",
                lat=34.6639,
                lon=135.5019,
                eta=datetime(2026, 9, 20, 14, 40),
                late=datetime(2026, 9, 20, 15, 10),
                people=2,
            )
        ],
    )


class LossModelTests(unittest.TestCase):
    def test_idle_origin_is_vehicle_gps(self):
        d = Driver("D2", 34.6639, 135.5019, datetime(2026, 9, 20, 14, 0), status="idle")
        lat, lon, t = d.dispatch_origin()
        self.assertAlmostEqual(lat, 34.6639)
        self.assertAlmostEqual(lon, 135.5019)
        self.assertEqual(t, d.free_at)

    def test_enroute_origin_is_destination_not_current_gps(self):
        d = Driver(
            "D3",
            34.6473,
            135.5139,
            datetime(2026, 9, 20, 14, 12),
            status="enroute",
            dest_lat=TRANSFER_STATION[0],
            dest_lon=TRANSFER_STATION[1],
        )
        lat, lon, t = d.dispatch_origin()
        self.assertAlmostEqual(lat, TRANSFER_STATION[0])
        self.assertAlmostEqual(lon, TRANSFER_STATION[1])
        self.assertEqual(t, datetime(2026, 9, 20, 14, 12))
        self.assertNotAlmostEqual(lat, d.lat)

    def test_argmin_is_nearer_idle_driver(self):
        group = _namba_songji()
        near = Driver("D2", 34.6639, 135.5019, datetime(2026, 9, 20, 14, 0))
        far = Driver("D4", 34.6654, 135.4323, datetime(2026, 9, 20, 14, 0))
        e_near = execute_group(group, *near.dispatch_origin()[:2], near.free_at, driver=near)
        e_far = execute_group(group, *far.dispatch_origin()[:2], far.free_at, driver=far)
        l_near = assignment_loss(e_near)
        l_far = assignment_loss(e_far)
        self.assertLess(l_near.total, l_far.total)
        self.assertLess(l_near.empty_min, l_far.empty_min)
        self.assertAlmostEqual(
            l_near.total,
            l_near.empty + l_near.detour + l_near.delay + l_near.wait + l_near.slack_risk + l_near.overtime,
            places=6,
        )

    def test_infeasible_is_infinite_loss(self):
        group = _namba_songji()
        group.stops[0].late = datetime(2026, 9, 20, 14, 1)
        far = Driver("D4", 34.6654, 135.4323, datetime(2026, 9, 20, 14, 0))
        exe = execute_group(group, far.lat, far.lon, far.free_at, driver=far)
        self.assertFalse(exe.feasible)
        self.assertGreaterEqual(assignment_loss(exe).total, 1e12)

    def test_unassigned_dominates_loss(self):
        d = Driver("D2", 34.6639, 135.5019, datetime(2026, 9, 20, 14, 0))
        g = _namba_songji()
        empty = Solution(plans=[DriverPlan(driver=d, groups=[])], unassigned=[g])
        assigned = Solution(plans=[DriverPlan(driver=d, groups=[g])], unassigned=[])
        self.assertGreater(solution_loss(empty).total, solution_loss(assigned).total)
        self.assertGreater(solution_loss(empty).miss, DEFAULT_WEIGHTS.unassigned_group - 1)
        self.assertIn("未派组", solution_loss(empty).explain())

    def test_explain_uses_accumulated_risk_not_slack_floor(self):
        d = Driver("D2", 34.6639, 135.5019, datetime(2026, 9, 20, 14, 0))
        g = _namba_songji()
        assigned = Solution(plans=[DriverPlan(driver=d, groups=[g])], unassigned=[])
        bd = solution_loss(assigned)
        self.assertAlmostEqual(bd.risk_raw, 0.0, places=5)
        self.assertIn("×0.0风险", bd.explain())

    def test_enroute_empty_starts_from_transfer(self):
        group = _namba_songji()
        d = Driver(
            "D3",
            34.6473,
            135.5139,
            datetime(2026, 9, 20, 14, 12),
            status="enroute",
            dest_lat=TRANSFER_STATION[0],
            dest_lon=TRANSFER_STATION[1],
        )
        olat, olon, t0 = d.dispatch_origin()
        exe = execute_group(group, olat, olon, t0, driver=d)
        self.assertTrue(exe.feasible)
        from_gps = execute_group(group, d.lat, d.lon, d.free_at, driver=d)
        self.assertNotAlmostEqual(exe.empty_min, from_gps.empty_min, places=2)

    def test_jieji_from_transfer_has_zero_empty(self):
        group = _namba_jieji()
        at_t = Driver("D3", TRANSFER_STATION[0], TRANSFER_STATION[1], datetime(2026, 9, 20, 14, 10))
        exe = execute_group(group, *at_t.dispatch_origin()[:2], at_t.free_at, driver=at_t)
        self.assertTrue(exe.feasible)
        self.assertLess(exe.empty_km, 0.05)
        self.assertAlmostEqual(exe.detour_km, 0.0, places=5)

    def test_jieji_from_hotel_pays_empty_back_to_t(self):
        group = _namba_jieji()
        at_hotel = Driver("D5", 34.6922, 135.4721, datetime(2026, 9, 20, 14, 0))
        at_t = Driver("D3", TRANSFER_STATION[0], TRANSFER_STATION[1], datetime(2026, 9, 20, 14, 0))
        e_hotel = execute_group(group, at_hotel.lat, at_hotel.lon, at_hotel.free_at, driver=at_hotel)
        e_t = execute_group(group, at_t.lat, at_t.lon, at_t.free_at, driver=at_t)
        self.assertTrue(e_hotel.feasible)
        self.assertGreater(e_hotel.empty_km, 3.0)
        self.assertLess(assignment_loss(e_t).total, assignment_loss(e_hotel).total)

    def test_l_counts_every_guest_not_only_first_stop(self):
        group = Group(
            gid="G",
            kind="to_station",
            stops=[
                Stop("A", "梅田A", 34.7055, 135.4983, datetime(2026, 9, 20, 14, 10), datetime(2026, 9, 20, 14, 40), 2),
                Stop("B", "梅田B", 34.7038, 135.5001, datetime(2026, 9, 20, 14, 16), datetime(2026, 9, 20, 14, 45), 1),
                Stop("C", "大阪站C", 34.7011, 135.4948, datetime(2026, 9, 20, 14, 22), datetime(2026, 9, 20, 14, 50), 3),
            ],
        )
        near = Driver("D1", 34.7024, 135.4959, datetime(2026, 9, 20, 14, 0))
        exe = execute_group(group, near.lat, near.lon, near.free_at, driver=near)
        self.assertTrue(exe.feasible)
        self.assertEqual(len(exe.stop_slacks), 3)
        self.assertGreater(exe.loaded_min, 0)
        self.assertGreater(exe.end_at, exe.arrive_first)
        first_only_wait, _, _ = (
            max(0.0, (group.first.eta - exe.arrive_first).total_seconds() / 60.0),
            None,
            None,
        )
        self.assertGreaterEqual(exe.hotel_wait_min, first_only_wait)

    def test_late_first_stop_cascades_delay_to_later_guests(self):
        group = Group(
            gid="G",
            kind="to_station",
            stops=[
                Stop("A", "A", 34.7055, 135.4983, datetime(2026, 9, 20, 14, 10), datetime(2026, 9, 20, 14, 50), 1),
                Stop("B", "B", 34.7038, 135.5001, datetime(2026, 9, 20, 14, 16), datetime(2026, 9, 20, 14, 55), 1),
            ],
        )
        # 人已在 A，但 14:20 才出发，首站晚 10 分钟，B 也会被推后
        late = Driver("D1", 34.7055, 135.4983, datetime(2026, 9, 20, 14, 20))
        exe = execute_group(group, late.lat, late.lon, late.free_at, driver=late, policy="immediate")
        self.assertTrue(exe.feasible)
        self.assertGreater(exe.eta_delay_min, 10.0)
        self.assertGreater(assignment_loss(exe).delay, assignment_loss(exe).empty)

    def test_later_stop_late_makes_group_infeasible(self):
        group = Group(
            gid="G",
            kind="to_station",
            stops=[
                Stop("A", "A", 34.7055, 135.4983, datetime(2026, 9, 20, 14, 10), datetime(2026, 9, 20, 14, 40), 1),
                Stop("B", "B", 34.6654, 135.4323, datetime(2026, 9, 20, 14, 12), datetime(2026, 9, 20, 14, 13), 1),
            ],
        )
        d = Driver("D1", 34.7055, 135.4983, datetime(2026, 9, 20, 14, 0))
        exe = execute_group(group, d.lat, d.lon, d.free_at, driver=d)
        self.assertFalse(exe.feasible)
        self.assertGreaterEqual(assignment_loss(exe).total, 1e12)


if __name__ == "__main__":
    unittest.main()
