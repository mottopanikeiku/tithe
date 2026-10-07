import gzip
import json

import pytest

from alignmenttax.io_utils import read_jsonl, scores_path


def test_compressed_scores_are_lossless(tmp_path):
    rows = [{"question_id": "q1", "raw_logprob_A": -1.23456789}, {"question_id": "q2"}]
    target = tmp_path / "scores.jsonl.gz"
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    assert scores_path(tmp_path) == target
    assert read_jsonl(target) == rows


def test_ambiguous_scores_are_rejected(tmp_path):
    (tmp_path / "scores.jsonl").write_text("{}\n")
    (tmp_path / "scores.jsonl.gz").write_bytes(gzip.compress(b"{}\n"))
    with pytest.raises(ValueError, match="Both compressed and plain"):
        scores_path(tmp_path)


def test_plain_score_paths_remain_supported(tmp_path):
    target = tmp_path / "scores.jsonl"
    target.write_text('{"question_id":"q"}\n')
    assert scores_path(tmp_path) == target
    assert scores_path(target) == target
    assert read_jsonl(target) == [{"question_id": "q"}]
