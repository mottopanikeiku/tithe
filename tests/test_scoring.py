from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alignmenttax import scoring

ITEM = {"id": "q1", "question": "Q?", "choices": {"A": "True", "B": "False"}, "correct_label": "A"}
CONFIG = {
    "models": {"base": {"model_id": "base", "revision": "base-sha"},
               "instruct": {"model_id": "instruct", "revision": "instruct-sha"}},
    "scoring": {"protocols": [scoring.SHARED_PLAIN_PROTOCOL, scoring.NATIVE_PROMPT_PROTOCOL]},
}


class ScoringTests(unittest.TestCase):
    def test_native_template_error_is_not_plain_fallback(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = ValueError("broken template")
        with self.assertRaisesRegex(ValueError, "broken template"):
            scoring.build_native_prompt(ITEM, model_key="instruct", tokenizer=tokenizer)
        with self.assertRaisesRegex(ValueError, "chat template"):
            scoring.build_native_prompt(ITEM, model_key="instruct")

    def test_resume_skips_before_loading_completed_model_and_scoring_row(self):
        completed = {(ITEM["id"], "base", p) for p in CONFIG["scoring"]["protocols"]}
        completed.add((ITEM["id"], "instruct", scoring.SHARED_PLAIN_PROTOCOL))
        scorer = Mock()
        scorer.model_id = "instruct"
        scorer.revision = "instruct-sha"
        scorer.device = "cpu"
        scorer.dtype = "bf16"
        scorer.tokenizer.apply_chat_template.return_value = "chat prompt"
        scorer.score_prompt.return_value = (-1.0, -2.0, {"A": 1, "B": 1})
        scorer.torch.cuda.is_available.return_value = False
        with patch.object(scoring, "TransformerLabelScorer", return_value=scorer) as load:
            rows = list(scoring.transformer_score_records([ITEM], config=CONFIG, existing_keys=completed))
        load.assert_called_once_with(model_key="instruct", model_config=CONFIG["models"]["instruct"])
        scorer.score_prompt.assert_called_once_with("chat prompt")
        self.assertEqual(rows[0]["prompt_format"], "chat_template")
        self.assertEqual(rows[0]["model_revision"], "instruct-sha")

    def test_completed_run_never_loads_model(self):
        completed = {(ITEM["id"], model, p) for model in CONFIG["models"]
                     for p in CONFIG["scoring"]["protocols"]}
        with patch.object(scoring, "TransformerLabelScorer") as load:
            self.assertEqual(list(scoring.transformer_score_records([ITEM], config=CONFIG, existing_keys=completed)), [])
        load.assert_not_called()

    def test_fake_resume_skips_computation_and_counts_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "data.jsonl"
            dataset.write_text(json.dumps(ITEM) + "\n")
            config = dict(CONFIG, dataset={"path": str(dataset)})
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            scores = root / "scores.jsonl"
            self.assertEqual(scoring.score_run(config_path=config_path, out=scores, fake=True), 4)
            with patch.object(scoring, "_fake_label_logprobs", side_effect=AssertionError("recomputed")):
                self.assertEqual(scoring.score_run(config_path=config_path, out=scores, fake=True, resume=True), 0)
            metadata = json.loads((root / "run_metadata.json").read_text())
            self.assertEqual(metadata["score_rows_skipped"], 4)
            self.assertEqual(metadata["score_rows_total"], 4)
            rows = [json.loads(line) for line in scores.read_text().splitlines()]
            self.assertEqual({row["prompt_format"] for row in rows}, {"synthetic"})

    def test_plain_prompt_format_is_recorded(self):
        scorer = Mock()
        scorer.model_id = "base"
        scorer.revision = "base-sha"
        scorer.device = "cpu"
        scorer.dtype = "bf16"
        scorer.score_prompt.return_value = (-1.0, -2.0, {})
        scorer.torch.cuda.is_available.return_value = False
        config = {"models": {"base": CONFIG["models"]["base"]}, "scoring": CONFIG["scoring"]}
        with patch.object(scoring, "TransformerLabelScorer", return_value=scorer):
            rows = list(scoring.transformer_score_records([ITEM], config=config))
        self.assertEqual([row["prompt_format"] for row in rows], ["plain", "plain"])


if __name__ == "__main__":
    unittest.main()
