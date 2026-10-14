"""
ChartQA prompt and metric for `multimodal/vision-language-chart-qa`.

Relaxed accuracy (Masry et al., 2022 - the official ChartQA metric):
    * numeric answers are correct within 5 % of the gold value
      (reading a value off a bar chart is never pixel-exact);
    * everything else must match exactly, ignoring case.
"""

# Asking for "a single word or number" keeps answers short and comparable.
INSTRUCTION = "Answer the question with a single word or number."


def chart_messages(ex, with_answer):
    """Chat messages for one ChartQA example.

    The {"type": "image"} entry is a placeholder: the processor replaces it
    with the chart's visual tokens. With `with_answer`, the first gold label is
    appended as the assistant turn (training); otherwise the prompt ends with
    the user turn (generation).
    """
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": f"{ex['query']}\n{INSTRUCTION}"}]}]
    if with_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": str(ex["label"][0])}]})
    return msgs


def _clean(s):
    """Normalise an answer: trim spaces and a final '.', drop ',', '%', '$', lower-case."""
    return str(s).strip().strip(".").replace(",", "").replace("%", "").replace("$", "").lower()


def relaxed_match(pred, gold, tol=0.05):
    """ChartQA relaxed accuracy for one answer: numbers within `tol` (5 %), otherwise exact match."""
    p, g = _clean(pred), _clean(gold)
    try:
        pf, gf = float(p), float(g)
        # Relative tolerance is undefined for a gold value of 0 -> require equality.
        return abs(pf - gf) <= tol * abs(gf) if gf != 0 else pf == gf
    except ValueError:
        # At least one side is not a number: compare the cleaned strings.
        return p == g
