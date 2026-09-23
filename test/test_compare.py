from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from fleet_mvp.assign import GroupALNS, greedy_assign
from fleet_mvp.compare_result import write_real_comparison
from fleet_mvp.config import TRANSFER_STATION
from fleet_mvp.export_result import write_dispatch_xlsx
from fleet_mvp.models import Driver, DriverPlan, Group, Solution, Stop


def _drivers():
    tlat, tlon = TRANSFER_STATION
    return [
        Driver("D1", tlat, tlon, datetime(2026, 9, 22, 8, 0), name="甲", plate="1001"),
        Driver("D2", 34.6639, 135.5019, datetime(2026, 9, 22, 8, 0), name="乙", plate="1002"),
    ]


def _groups():
    return [
        Group(
            gid="G1",
            kind="to_station",
            note="历史司机甲【1001】",
            stops=[
                Stop("S1", "难波", 34.6639, 135.5019, datetime(2026, 9, 22, 9, 0), datetime(2026, 9, 22, 9, 20), 2, guest="客人甲")
            ],
        ),
        Group(
            gid="G2",
            kind="from_station",
            note="历史司机乙【1002】",
            stops=[
                Stop("S2", "难波", 34.6642, 135.5036, datetime(2026, 9, 22, 11, 0), datetime(2026, 9, 22, 11, 20), 1, guest="客人乙")
            ],
        ),
    ]


class CompareReportTests(unittest.TestCase):
    def test_markdown_uses_the_same_stem_as_the_workbook(self):
        drivers, groups = _drivers(), _groups()
        now = datetime(2026, 9, 22, 8, 0)
        greedy = greedy_assign(drivers, groups, now=now)
        alns = GroupALNS(drivers, groups, max_iter=5, seed=1, now=now).run(verbose=False)
        with tempfile.TemporaryDirectory() as folder:
            stem = Path(folder) / "运力派单-20260923-120000"
            xlsx = write_dispatch_xlsx(alns, now=now, path=stem.with_suffix(".xlsx"))
            md = write_real_comparison(
                stem.with_suffix(".md"),
                drivers,
                groups,
                greedy,
                alns,
                now,
                iters=5,
                seed=1,
                xlsx_name=xlsx.name,
                xlsx_mode="ALNS",
            )
            self.assertEqual(xlsx.stem, md.stem)
            text = md.read_text(encoding="utf-8")
            self.assertIn("# 0922：线上、贪心、ALNS", text)
            self.assertIn("这份表格是 ALNS", text)
            self.assertIn(xlsx.name, text)
            self.assertIn("| 司机 | 线上 | 贪心 | ALNS |", text)
        self.assertIn("甲", text)
        self.assertIn("乙", text)

    def test_infeasible_solution_does_not_write_an_empty_workbook(self):
        driver = Driver("D9", 34.66, 135.50, datetime(2026, 9, 22, 8, 0), on_shift=False)
        group = _groups()[0]
        sol = Solution(plans=[DriverPlan(driver=driver, groups=[group])], unassigned=[])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "运力派单.xlsx"
            with self.assertRaises(ValueError):
                write_dispatch_xlsx(sol, path=path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
