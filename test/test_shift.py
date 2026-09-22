from __future__ import annotations

import unittest
from datetime import datetime

from fleet_mvp.assign import greedy_assign
from fleet_mvp.execute import execute_for_driver
from fleet_mvp.loss import assignment_loss
from fleet_mvp.models import Driver, Group, Stop


def _songji() -> Group:
    return Group(
        gid="G02",
        kind="to_station",
        stops=[
            Stop("B01", "难波A", 34.6628, 135.5028, datetime(2026, 9, 20, 8, 8), datetime(2026, 9, 20, 8, 30), 2),
            Stop("B02", "心斋桥", 34.6751, 135.5012, datetime(2026, 9, 20, 8, 18), datetime(2026, 9, 20, 8, 45), 2),
        ],
    )


def _jieji_evening() -> Group:
    return Group(
        gid="J32",
        kind="from_station",
        stops=[
            Stop("B31", "难波C", 34.6642, 135.5036, datetime(2026, 9, 20, 18, 5), datetime(2026, 9, 20, 18, 30), 2),
        ],
    )


class ShiftGateTests(unittest.TestCase):
    def test_off_shift_driver_is_not_considered(self):
        group = _songji()
        off = Driver("D9", 34.6639, 135.5019, datetime(2026, 9, 20, 8, 0), name="休息", on_shift=False)
        on = Driver(
            "D2",
            34.6639,
            135.5019,
            datetime(2026, 9, 20, 8, 0),
            name="佐藤",
            on_shift=True,
            shift_start=datetime(2026, 9, 20, 8, 0),
            shift_end=datetime(2026, 9, 20, 21, 0),
        )
        self.assertFalse(off.consider_for(group))
        self.assertTrue(on.consider_for(group))
        exe = execute_for_driver(group, off)
        self.assertFalse(exe.feasible)
        self.assertGreaterEqual(assignment_loss(exe).total, 1e12)

    def test_morning_shift_excluded_from_evening_group(self):
        group = _jieji_evening()
        morning = Driver(
            "D1",
            34.7024,
            135.4959,
            datetime(2026, 9, 20, 8, 0),
            shift_start=datetime(2026, 9, 20, 8, 0),
            shift_end=datetime(2026, 9, 20, 16, 0),
        )
        night = Driver(
            "D8",
            34.6614,
            135.4887,
            datetime(2026, 9, 20, 17, 0),
            shift_start=datetime(2026, 9, 20, 17, 0),
            shift_end=datetime(2026, 9, 20, 22, 0),
        )
        self.assertFalse(morning.consider_for(group))
        self.assertTrue(night.consider_for(group))

    def test_greedy_skips_off_duty_and_assigns_on_duty(self):
        group = _songji()
        off = Driver("D9", 34.6639, 135.5019, datetime(2026, 9, 20, 8, 0), on_shift=False)
        on = Driver("D2", 34.6639, 135.5019, datetime(2026, 9, 20, 8, 0))
        sol = greedy_assign([off, on], [group])
        d2 = next(p for p in sol.plans if p.driver.did == "D2")
        d9 = next(p for p in sol.plans if p.driver.did == "D9")
        self.assertEqual([g.gid for g in d2.groups], ["G02"])
        self.assertEqual(d9.groups, [])

    def test_trip_past_shift_end_is_overtime_not_rejected(self):
        group = _songji()
        short = Driver(
            "D7",
            34.6639,
            135.5019,
            datetime(2026, 9, 20, 8, 0),
            shift_start=datetime(2026, 9, 20, 8, 0),
            shift_end=datetime(2026, 9, 20, 8, 20),
        )
        self.assertTrue(short.consider_for(group))
        exe = execute_for_driver(group, short)
        self.assertTrue(exe.feasible)
        self.assertTrue(exe.overtime)
        self.assertGreater(exe.overtime_min, 0)
        self.assertIn("加班", exe.reason)
        self.assertIn("加班", assignment_loss(exe).explain())

    def test_on_time_driver_beats_overtime_when_both_can_take(self):
        group = _songji()
        short = Driver(
            "D7",
            34.6639,
            135.5019,
            datetime(2026, 9, 20, 8, 0),
            shift_end=datetime(2026, 9, 20, 8, 20),
        )
        full = Driver(
            "D2",
            34.6639,
            135.5019,
            datetime(2026, 9, 20, 8, 0),
            shift_end=datetime(2026, 9, 20, 21, 0),
        )
        sol = greedy_assign([short, full], [group])
        d2 = next(p for p in sol.plans if p.driver.did == "D2")
        d7 = next(p for p in sol.plans if p.driver.did == "D7")
        self.assertEqual([g.gid for g in d2.groups], ["G02"])
        self.assertEqual(d7.groups, [])


if __name__ == "__main__":
    unittest.main()
