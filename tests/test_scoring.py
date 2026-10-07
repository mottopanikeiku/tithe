from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    def test_configured_template_date_reaches_native_rendering(self):
        config = {"models": {"instruct": CONFIG["models"]["instruct"]},
                  "scoring": {"protocols": [scoring.NATIVE_PROMPT_PROTOCOL],
                              "template_date": "2026-10-07"}}
        scorer = Mock()
        scorer.model_id, scorer.revision = "instruct", "instruct-sha"
        scorer.device, scorer.dtype = "cpu", "bf16"
        scorer.tokenizer.apply_chat_template.return_value = "dated prompt"
        scorer.score_prompt.return_value = (-1.0, -2.0, {"A": 1, "B": 1})
        scorer.torch.cuda.is_available.return_value = False
        with patch.object(scoring, "TransformerLabelScorer", return_value=scorer):
            rows = list(scoring.transformer_score_records([ITEM], config=config))
        self.assertEqual(len(rows), 1)
        helper = scorer.tokenizer.apply_chat_template.call_args.kwargs["strftime_now"]
        self.assertEqual(helper("%Y-%m-%d"), "2026-10-07")

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

    def test_batched_resume_preserves_ids_protocols_and_provenance(self):
        items = [
            dict(ITEM, id="q-long", question="A much longer question?"),
            dict(ITEM, id="q-short", question="Q?"),
            dict(ITEM, id="q-medium", question="Medium question?"),
        ]
        config = dict(CONFIG, scoring=dict(CONFIG["scoring"], batch_size=2))
        completed = {(item["id"], "base", p) for item in items for p in CONFIG["scoring"]["protocols"]}
        completed.add(("q-short", "instruct", scoring.SHARED_PLAIN_PROTOCOL))
        scorer = self._mock_scorer("instruct")
        scorer.tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: messages[0]["content"]
        scorer.score_prompts.side_effect = lambda prompts: [
            (-float(len(prompt)), -2.0, {"A": 1, "B": 2}) for prompt in prompts
        ]
        with patch.object(scoring, "TransformerLabelScorer", return_value=scorer) as load:
            rows = list(scoring.transformer_score_records(iter(items), config=config, existing_keys=completed))
        load.assert_called_once_with(model_key="instruct", model_config=CONFIG["models"]["instruct"])
        scorer.score_prompt.assert_not_called()
        self.assertEqual([len(call.args[0]) for call in scorer.score_prompts.call_args_list], [2, 2, 1])
        self.assertEqual(len(rows), 5)
        for row in rows:
            item = next(item for item in items if item["id"] == row["question_id"])
            prompt = scoring.build_prompt(
                item, protocol=row["prompt_protocol"], model_key="instruct", tokenizer=scorer.tokenizer,
            )
            self.assertEqual(row["raw_logprob_A"], -len(prompt))
            self.assertEqual(row["question"], item["question"])
            self.assertEqual(row["model_revision"], "instruct-sha")
            self.assertEqual(row["label_token_counts"], {"A": 1, "B": 2})
            self.assertGreaterEqual(row["elapsed_seconds"], 0.0)
            self.assertEqual(
                row["prompt_format"],
                "chat_template" if row["prompt_protocol"] == scoring.NATIVE_PROMPT_PROTOCOL else "plain",
            )

    def test_default_scalar_path_keeps_prompt_building_lazy(self):
        items = [ITEM, dict(ITEM, id="q2")]
        config = {
            "models": {"instruct": CONFIG["models"]["instruct"]},
            "scoring": {"protocols": [scoring.NATIVE_PROMPT_PROTOCOL]},
        }
        scorer = self._mock_scorer("instruct")
        scorer.tokenizer.apply_chat_template.side_effect = ["first prompt", ValueError("second template")]
        with patch.object(scoring, "TransformerLabelScorer", return_value=scorer):
            rows = scoring.transformer_score_records(items, config=config)
            self.assertEqual(next(rows)["question_id"], ITEM["id"])
            scorer.score_prompt.assert_called_once_with("first prompt")
            scorer.score_prompts.assert_not_called()
            with self.assertRaisesRegex(ValueError, "second template"):
                next(rows)

    def test_batched_completed_run_never_loads_model(self):
        completed = {(ITEM["id"], model, p) for model in CONFIG["models"]
                     for p in CONFIG["scoring"]["protocols"]}
        config = dict(CONFIG, scoring=dict(CONFIG["scoring"], batch_size=4))
        with patch.object(scoring, "TransformerLabelScorer") as load:
            self.assertEqual(list(scoring.transformer_score_records([ITEM], config=config, existing_keys=completed)), [])
        load.assert_not_called()

    def test_real_file_resume_create_append_complete_and_overwrite(self):
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                items = [dict(ITEM, id=f"q{i}", question=f"Question {i}?") for i in range(3)]
                dataset = root / "data.jsonl"
                dataset.write_text("".join(json.dumps(item) + "\n" for item in items))
                config = {
                    "models": {"base": CONFIG["models"]["base"]},
                    "scoring": dict(CONFIG["scoring"], batch_size=batch_size),
                    "dataset": {"path": str(dataset)},
                }
                config_path = root / "config.json"
                config_path.write_text(json.dumps(config))
                scores = root / "scores.jsonl"
                scorer = self._mock_scorer("base")
                with patch.object(scoring, "TransformerLabelScorer", return_value=scorer):
                    self.assertEqual(scoring.score_run(
                        config_path=config_path, out=scores, limit=1, resume=True,
                    ), 2)
                    prefix = scores.read_text()
                    self.assertEqual(scoring.score_run(config_path=config_path, out=scores, resume=True), 4)
                    self.assertTrue(scores.read_text().startswith(prefix))
                rows = [json.loads(line) for line in scores.read_text().splitlines()]
                self.assertEqual(len(rows), 6)
                self.assertEqual(len({scoring._score_key(row) for row in rows}), 6)
                metadata = json.loads((root / "run_metadata.json").read_text())
                self.assertEqual(metadata["score_rows_skipped"], 2)
                self.assertEqual(metadata["score_rows_written"], 4)
                self.assertEqual(metadata["score_rows_total"], 6)
                complete_text = scores.read_text()
                with patch.object(scoring, "TransformerLabelScorer") as load:
                    self.assertEqual(scoring.score_run(config_path=config_path, out=scores, resume=True), 0)
                load.assert_not_called()
                self.assertEqual(scores.read_text(), complete_text)
                with patch.object(scoring, "TransformerLabelScorer", return_value=scorer):
                    self.assertEqual(scoring.score_run(config_path=config_path, out=scores), 6)
                self.assertEqual(len(scores.read_text().splitlines()), 6)

    def test_fake_partial_resume_creates_and_appends_without_rescoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items = [ITEM, dict(ITEM, id="q2")]
            dataset = root / "data.jsonl"
            dataset.write_text("".join(json.dumps(item) + "\n" for item in items))
            config_path = root / "config.json"
            config_path.write_text(json.dumps(dict(CONFIG, dataset={"path": str(dataset)})))
            scores = root / "scores.jsonl"
            self.assertEqual(scoring.score_run(
                config_path=config_path, out=scores, fake=True, resume=True, limit=1,
            ), 4)
            original = scoring._fake_label_logprobs

            def only_missing(item, **kwargs):
                self.assertEqual(item["id"], "q2")
                return original(item, **kwargs)

            with patch.object(scoring, "_fake_label_logprobs", side_effect=only_missing) as compute:
                self.assertEqual(scoring.score_run(
                    config_path=config_path, out=scores, fake=True, resume=True,
                ), 4)
            self.assertEqual(compute.call_count, 4)
            rows = [json.loads(line) for line in scores.read_text().splitlines()]
            self.assertEqual(len({scoring._score_key(row) for row in rows}), 8)
            metadata = json.loads((root / "run_metadata.json").read_text())
            self.assertEqual(metadata["score_rows_written"], 4)
            self.assertEqual(metadata["score_rows_skipped"], 4)
            self.assertEqual(metadata["score_rows_total"], 8)

    def test_live_resume_keys_skip_duplicates_within_and_between_batches(self):
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                config = {
                    "models": {"base": CONFIG["models"]["base"]},
                    "scoring": {"protocols": [scoring.SHARED_PLAIN_PROTOCOL], "batch_size": batch_size},
                }
                keys = set()
                scorer = self._mock_scorer("base")
                with patch.object(scoring, "TransformerLabelScorer", return_value=scorer):
                    iterator = scoring.transformer_score_records([ITEM] * 4, config=config, existing_keys=keys)
                    row = next(iterator)
                    keys.add(scoring._score_key(row))
                    self.assertEqual(list(iterator), [])

    def test_invalid_batch_size_rejected_before_loading(self):
        for batch_size in (0, -1, 1.5, "2", True):
            with self.subTest(batch_size=batch_size), patch.object(scoring, "TransformerLabelScorer") as load:
                config = dict(CONFIG, scoring=dict(CONFIG["scoring"], batch_size=batch_size))
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    list(scoring.transformer_score_records([ITEM], config=config))
                load.assert_not_called()

    def test_loading_honors_sdpa_and_detects_explicit_logits_support(self):
        for argument in ("logits_to_keep", "num_logits_to_keep", None):
            with self.subTest(argument=argument):
                torch = Mock()
                torch.cuda.is_available.return_value = False
                model = Mock()
                parameter = SimpleNamespace(device="cpu", dtype="torch.bfloat16")
                model.parameters.side_effect = lambda: iter([parameter])
                forwards = {
                    "logits_to_keep": lambda input_ids, logits_to_keep=0: None,
                    "num_logits_to_keep": lambda input_ids, num_logits_to_keep=0: None,
                    None: lambda input_ids, **kwargs: None,
                }
                model.forward = forwards[argument]
                tokenizer = Mock()
                tokenizer.encode.side_effect = [[3], [4]]
                transformers = Mock()
                transformers.AutoModelForCausalLM.from_pretrained.return_value = model
                transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
                model_config = dict(CONFIG["models"]["base"], dtype="bf16", attn_implementation="sdpa")
                with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
                    scorer = scoring.TransformerLabelScorer(model_key="base", model_config=model_config)
                kwargs = transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
                self.assertEqual(kwargs["attn_implementation"], "sdpa")
                self.assertEqual(kwargs["revision"], "base-sha")
                self.assertIs(kwargs["torch_dtype"], torch.bfloat16)
                self.assertEqual(scorer.logits_keep_argument, argument)

    @staticmethod
    def _mock_scorer(model_key):
        scorer = Mock()
        scorer.model_id = model_key
        scorer.revision = f"{model_key}-sha"
        scorer.device = "cpu"
        scorer.dtype = "bf16"
        scorer.score_prompt.return_value = (-1.0, -2.0, {"A": 1, "B": 1})
        scorer.score_prompts.side_effect = lambda prompts: [
            (-1.0, -2.0, {"A": 1, "B": 1}) for _ in prompts
        ]
        scorer.torch.cuda.is_available.return_value = False
        return scorer


class BatchedLikelihoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Numerical batching tests require optional torch.")
        cls.torch = torch

    def _scorer(self, *, labels, padding_side, keep_argument="logits_to_keep", pad_id=0):
        torch = self.torch

        class Tokenizer:
            eos_token_id = None
            pad_token_id = pad_id

            def encode(self, text, add_special_tokens=False):
                if text in labels:
                    return list(labels[text])
                # Concatenated text intentionally tokenizes differently: the
                # tested scorer must tokenize prompt and labels separately.
                return [ord(character) % 13 + 1 for character in text]

            def __call__(self, text, **kwargs):
                ids = torch.tensor([self.encode(text)], dtype=torch.long)
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        class Model:
            def __init__(self):
                self.calls = []

            def logits(self, input_ids, attention_mask, position_ids, keep):
                if position_ids is None:
                    position_ids = torch.arange(input_ids.shape[1])[None, :]
                else:
                    expected = (attention_mask.cumsum(1) - 1).clamp_min(0)
                    torch.testing.assert_close(position_ids, expected)
                self.calls.append({
                    "input_ids": input_ids.clone(), "attention_mask": attention_mask.clone(),
                    "position_ids": position_ids.clone(), "keep": keep,
                })
                context = (input_ids * attention_mask).cumsum(1)
                center = ((context + 3 * position_ids) % 17).float()
                vocab = torch.arange(19, dtype=torch.float32)
                logits = (-0.37 * (vocab - center[:, :, None]).abs() + vocab * 0.013).to(torch.bfloat16)
                if keep:
                    logits = logits[:, -keep:, :]
                return SimpleNamespace(logits=logits)

            def __call__(self, **kwargs):
                return self.forward(**kwargs)

        class CurrentModel(Model):
            def forward(self, input_ids, attention_mask, position_ids=None, use_cache=False, logits_to_keep=0):
                return self.logits(input_ids, attention_mask, position_ids, logits_to_keep)

        class OlderModel(Model):
            def forward(self, input_ids, attention_mask, position_ids=None, use_cache=False, num_logits_to_keep=0):
                return self.logits(input_ids, attention_mask, position_ids, num_logits_to_keep)

        class FullLogitsModel(Model):
            def forward(self, input_ids, attention_mask, position_ids=None, use_cache=False):
                return self.logits(input_ids, attention_mask, position_ids, 0)

        scorer = scoring.TransformerLabelScorer.__new__(scoring.TransformerLabelScorer)
        scorer.torch = torch
        scorer.device = "cpu"
        scorer.tokenizer = Tokenizer()
        scorer.tokenizer.padding_side = padding_side
        model_type = {
            "logits_to_keep": CurrentModel, "num_logits_to_keep": OlderModel, None: FullLogitsModel,
        }[keep_argument]
        scorer.model = model_type()
        scorer.logits_keep_argument = next(
            (name for name in ("logits_to_keep", "num_logits_to_keep")
             if name in inspect.signature(scorer.model.forward).parameters), None,
        )
        scorer.label_token_ids = {label: scorer.tokenizer.encode(text) for label, text in scoring.LABEL_TEXT.items()}
        return scorer

    def _reference(self, scorer, prompt):
        torch = self.torch
        values = []
        for label, text in scoring.LABEL_TEXT.items():
            prompt_ids = scorer.tokenizer.encode(prompt)
            target_ids = scorer.tokenizer.encode(text)
            ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long)
            logits = scorer.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            values.append(sum(float(log_probs[0, len(prompt_ids) + offset - 1, token_id])
                              for offset, token_id in enumerate(target_ids)))
        return values

    def test_batch_matches_independent_reference_and_scalar_for_all_padding_and_label_lengths(self):
        prompts = ["x", "longer prompt", "mid"]
        for labels in ({" A": [3], " B": [5]}, {" A": [3, 8], " B": [5, 9, 2]}, {" A": [3], " B": [5, 9]}):
            for padding in ("left", "right"):
                for keep in ("logits_to_keep", "num_logits_to_keep", None):
                    with self.subTest(labels=labels, padding=padding, keep=keep):
                        scorer = self._scorer(labels=labels, padding_side=padding, keep_argument=keep)
                        reference = [self._reference(scorer, prompt) for prompt in prompts]
                        scalar = [scorer.score_prompt(prompt) for prompt in prompts]
                        scorer.model.calls.clear()
                        batch = scorer.score_prompts(prompts)
                        self.assertEqual(len(scorer.model.calls), 1)
                        self.assertEqual(scorer.model.calls[0]["input_ids"].shape[0],
                                         len(prompts) if all(len(ids) == 1 for ids in labels.values()) else 2 * len(prompts))
                        for expected, single, actual in zip(reference, scalar, batch, strict=True):
                            self.assertEqual(actual[2], {label: len(labels[text]) for label, text in scoring.LABEL_TEXT.items()})
                            for index in (0, 1):
                                self.assertAlmostEqual(actual[index], expected[index], places=6)
                                self.assertAlmostEqual(actual[index], single[index], places=6)
                        if keep is not None and padding == "left" and all(len(ids) == 1 for ids in labels.values()):
                            self.assertEqual(scorer.model.calls[0]["keep"], 1)

    def test_fp32_softmax_only_receives_required_prediction_positions(self):
        scorer = self._scorer(labels={" A": [3], " B": [5]}, padding_side="left")
        original = self.torch.log_softmax
        with patch.object(self.torch, "log_softmax", wraps=original) as softmax:
            scorer.score_prompts(["x", "longer prompt"])
        softmax.assert_called_once()
        selected = softmax.call_args.args[0]
        self.assertEqual(selected.dtype, self.torch.float32)
        self.assertEqual(tuple(selected.shape), (2, 19))

    def test_no_pad_token_uses_masked_in_vocabulary_fallback(self):
        for padding in ("left", "right"):
            with self.subTest(padding=padding):
                scorer = self._scorer(labels={" A": [3, 8], " B": [5]}, padding_side=padding, pad_id=None)
                scalar = [scorer.score_prompt(prompt) for prompt in ("x", "longer")]
                batch = scorer.score_prompts(["x", "longer"])
                self.assertEqual(batch, scalar)
                self.assertIsNone(scorer.tokenizer.pad_token_id)

    def test_empty_batch_empty_prompt_and_empty_label(self):
        scorer = self._scorer(labels={" A": [3], " B": [5]}, padding_side="right")
        self.assertEqual(scorer.score_prompts([]), [])
        self.assertEqual(scorer.model.calls, [])
        with self.assertRaisesRegex(ValueError, "Prompt tokenization"):
            scorer.score_prompts([""])
        scorer.label_token_ids["A"] = []
        with self.assertRaisesRegex(ValueError, "Continuation tokenization"):
            scorer.score_prompts(["x"])


if __name__ == "__main__":
    unittest.main()
