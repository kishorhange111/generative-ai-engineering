"""
Speech-recognition metrics for `multimodal/whisper-marathi-speech-recognition`.

    WER = (substitutions + deletions + insertions) / number of reference words
    CER = the same, counted on characters

Both are *corpus-level*: edits are summed over all utterances and divided by
the total reference length (not an average of per-utterance rates). WER can
exceed 100 % when the model inserts many extra words.

We use `jiwer`, the standard implementation, so numbers are comparable with
other published Whisper fine-tuning results.
"""


def wer_cer(refs, hyps):
    """Corpus-level (WER, CER) of hypotheses `hyps` against references `refs` (lists of strings)."""
    import jiwer  # imported lazily: only needed when a metric is actually computed

    return float(jiwer.wer(refs, hyps)), float(jiwer.cer(refs, hyps))
