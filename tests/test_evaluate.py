import json
import tempfile
import unittest
from pathlib import Path

from evaluate import (
    DEFAULT_AUTO_REPLIES,
    DEFAULT_HUMAN_REF,
    DEFAULT_MOCK_RESPONSES,
    InputError,
    METRIC_DEFINITIONS,
    MetricResult,
    MockJudgeClient,
    build_cases,
    evaluate_dataset,
    load_json_array,
    parse_judge_response,
    score_band,
    weighted_score,
)


class EvaluateTest(unittest.TestCase):
    def test_metric_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(
            sum(item["weight"] for item in METRIC_DEFINITIONS.values()), 1.0
        )

    def test_weighted_score_uses_confirmed_rubric(self) -> None:
        metrics = {
            "reference_consistency": MetricResult(3, "reason"),
            "intent_completion": MetricResult(2, "reason"),
            "actionability": MetricResult(1, "reason"),
            "tone": MetricResult(3, "reason"),
        }
        self.assertEqual(weighted_score(metrics), 52.5)

    def test_score_bands(self) -> None:
        self.assertEqual(score_band(85), "优秀")
        self.assertEqual(score_band(70), "合格")
        self.assertEqual(score_band(50), "需要改进")
        self.assertEqual(score_band(49.9), "不合格")

    def test_placeholder_does_not_invalidate_case(self) -> None:
        cases = build_cases(
            [{"id": "case", "user_question": "成分？", "auto_reply": "请查看详情"}],
            [{"id": "case", "human_reference": "主要成分是XX", "annotator_notes": ""}],
        )
        self.assertEqual(len(cases), 1)
        self.assertTrue(cases[0].reference_has_placeholder)

    def test_mismatched_ids_are_rejected(self) -> None:
        replies = [{"id": "a", "user_question": "问题", "auto_reply": "回复"}]
        refs = [{"id": "b", "human_reference": "参考", "annotator_notes": ""}]
        with self.assertRaises(InputError):
            build_cases(replies, refs)

    def test_invalid_mock_score_is_rejected(self) -> None:
        response = {
            "metrics": {
                key: {"score": 5 if key == "tone" else 4, "reason": "reason"}
                for key in METRIC_DEFINITIONS
            },
            "missing_points": [],
            "risk_flags": [],
            "summary": "summary",
        }
        with self.assertRaises(InputError):
            parse_judge_response(json.dumps(response))

    def test_full_mock_pipeline(self) -> None:
        judge = MockJudgeClient(DEFAULT_MOCK_RESPONSES)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            payload = evaluate_dataset(
                DEFAULT_AUTO_REPLIES, DEFAULT_HUMAN_REF, output_dir, judge
            )
            self.assertEqual(payload["summary"]["case_count"], 20)
            self.assertEqual(payload["summary"]["overall_mean"], 71.1)
            self.assertEqual(
                payload["summary"]["band_distribution"],
                {"需要改进": 7, "合格": 8, "优秀": 3, "不合格": 2},
            )
            placeholders = [
                row["id"]
                for row in payload["results"]
                if row["reference_has_placeholder"]
            ]
            self.assertEqual(placeholders, ["case_13"])
            self.assertEqual(
                payload["run"]["input"], "data/task3_auto_replies.json"
            )
            self.assertEqual(
                payload["run"]["human_reference"], "data/task3_human_ref.json"
            )
            self.assertNotIn(str(Path.home()), json.dumps(payload, ensure_ascii=False))
            self.assertTrue((output_dir / "evaluation_results.json").exists())
            self.assertTrue((output_dir / "evaluation_report.md").exists())

    def test_source_data_are_valid_json_arrays(self) -> None:
        self.assertEqual(len(load_json_array(DEFAULT_AUTO_REPLIES)), 20)
        self.assertEqual(len(load_json_array(DEFAULT_HUMAN_REF)), 20)


if __name__ == "__main__":
    unittest.main()
