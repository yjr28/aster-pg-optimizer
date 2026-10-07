import unittest

from aster_ranker import CandidatePlan, rank_plans


class RankerTests(unittest.TestCase):
    def test_selects_clear_low_score_plan(self):
        plans = [
            CandidatePlan("postgres_default", 1200, 50000, seq_scans=2, uncertainty=0.10),
            CandidatePlan("index_join", 180, 2500, seq_scans=0, uncertainty=0.08),
            CandidatePlan("nested_loop", 700, 12000, nested_loops=3, uncertainty=0.12),
        ]
        decision = rank_plans(plans, postgres_choice="postgres_default")
        self.assertEqual(decision.selected_plan, "index_join")
        self.assertFalse(decision.fallback)

    def test_high_uncertainty_falls_back_to_postgres(self):
        plans = [
            CandidatePlan("postgres_default", 600, 9000, uncertainty=0.05),
            CandidatePlan("risky_model_pick", 90, 1000, uncertainty=0.90),
        ]
        decision = rank_plans(plans, postgres_choice="postgres_default")
        self.assertTrue(decision.fallback)
        self.assertEqual(decision.selected_plan, "postgres_default")
        self.assertIn("uncertainty", decision.reason)

    def test_close_ranking_falls_back(self):
        plans = [
            CandidatePlan("postgres_default", 100, 1000, uncertainty=0.10),
            CandidatePlan("candidate_b", 101, 1000, uncertainty=0.10),
        ]
        decision = rank_plans(plans, postgres_choice="postgres_default", min_margin=0.08)
        self.assertTrue(decision.fallback)
        self.assertEqual(decision.selected_plan, "postgres_default")


if __name__ == "__main__":
    unittest.main()
