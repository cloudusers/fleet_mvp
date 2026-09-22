from __future__ import annotations

import unittest

from fleet_mvp.config import KIND_FROM_STATION, KIND_TO_STATION
from fleet_mvp.io_util import data_dir, default_dataset, load_drivers


class DatasetTests(unittest.TestCase):
    def test_drivers_have_name_plate_and_enroute_dest(self):
        drivers = load_drivers(data_dir() / "drivers.json")
        self.assertEqual(len(drivers), 5)
        self.assertEqual(len({d.did for d in drivers}), 5)
        for d in drivers:
            self.assertTrue(d.name)
            self.assertTrue(d.plate)
            self.assertTrue(d.on_shift)
            self.assertIsNotNone(d.shift_start)
            self.assertIsNotNone(d.shift_end)
        enroute = [d for d in drivers if d.status == "enroute"]
        self.assertGreaterEqual(len(enroute), 1)
        for d in enroute:
            self.assertIsNotNone(d.dest_lat)
            self.assertIsNotNone(d.dest_lon)
            self.assertTrue(d.current_gid)
            self.assertIn(d.current_kind, (KIND_TO_STATION, KIND_FROM_STATION))

    def test_one_dataset_covers_both_kinds_and_day_span(self):
        _drivers, groups = default_dataset()
        kinds = {g.kind for g in groups}
        self.assertEqual(kinds, {KIND_TO_STATION, KIND_FROM_STATION})
        hours = {g.first.eta.hour for g in groups}
        self.assertTrue(any(h < 12 for h in hours), msg="含早晨送机")
        self.assertTrue(any(12 <= h < 17 for h in hours), msg="含白天轮换")
        self.assertTrue(any(h >= 17 for h in hours), msg="含晚上接机")
        self._assert_groups_ok(groups, "default")

        _drivers, peak = default_dataset(peak=True)
        self._assert_groups_ok(peak, "peak")
        self.assertTrue(all(g.kind == KIND_TO_STATION for g in peak))

        drivers, trap = default_dataset(trap=True)
        self._assert_groups_ok(trap, "trap")
        self.assertEqual(len(drivers), 2)
        self.assertEqual({g.gid for g in trap}, {"T01", "T02"})

    def _assert_groups_ok(self, groups, label: str):
        gids = [g.gid for g in groups]
        self.assertEqual(len(gids), len(set(gids)), msg=label)
        for g in groups:
            self.assertTrue(g.stops, msg=g.gid)
            if g.is_to_station():
                self.assertTrue(g.route_text().endswith("-T"), msg=g.gid)
            else:
                self.assertTrue(g.route_text().startswith("T-"), msg=g.gid)
            prev = None
            for s in g.stops:
                self.assertTrue(s.name, msg=f"{g.gid}/{s.oid}")
                self.assertLess(s.eta, s.late, msg=f"{g.gid}/{s.oid}")
                self.assertGreaterEqual(s.people, 1)
                if prev is not None:
                    self.assertLessEqual(prev, s.eta, msg=f"{g.gid} 站序时间应非降")
                prev = s.eta


if __name__ == "__main__":
    unittest.main()
