"""Tests for the corpus-level WER / CER wrapper."""

import pytest

pytest.importorskip("jiwer")

from genai_common.speech import wer_cer  # noqa: E402


def test_perfect_transcription():
    assert wer_cer(["नमस्कार जग"], ["नमस्कार जग"]) == (0.0, 0.0)


def test_word_substitution():
    wer, cer = wer_cer(["the cat sat"], ["the cat sit"])
    assert wer == pytest.approx(1 / 3)
    assert cer == pytest.approx(1 / 11)


def test_insertions_can_exceed_100_percent():
    wer, _ = wer_cer(["hello"], ["hello hello hello"])
    assert wer == pytest.approx(2.0)


def test_corpus_level_not_mean_of_utterances():
    # Per-utterance WERs are 1/1 and 0/3 (mean 0.5); corpus-level is 1 error / 4 words = 0.25.
    wer, _ = wer_cer(["a", "b c d"], ["x", "b c d"])
    assert wer == pytest.approx(0.25)
