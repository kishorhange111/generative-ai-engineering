"""
GSM8K parsing and the *verifiable* reward functions used for GRPO training
(`reinforcement-learning/grpo-math-reasoning`).

"Reinforcement learning with verifiable rewards" (RLVR) means the reward is
computed by code, not by a learned reward model or a human:

    correctness_reward : +1.0 if the final number equals the ground truth
    format_reward      : +0.5 if the output is  <think>...</think><answer>...</answer>

Because these functions decide what the model is pushed towards, they are the
most important code in an RL pipeline - and the easiest to unit-test.
"""

import re

# The system prompt asks for a visible chain of thought followed by a short,
# machine-readable final answer. The same prompt is used before and after RL.
SYSTEM_PROMPT = (
    "You are a careful math tutor. First think step by step inside <think> </think> tags, "
    "then give only the final numeric answer inside <answer> </answer> tags."
)

# The whole completion must be exactly one <think> block followed by one
# <answer> block (only whitespace around them). re.S lets "." match newlines,
# because reasoning spans many lines. ".+?" = non-empty, non-greedy.
FORMAT_RE = re.compile(r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$", re.S)

# A number: optional minus, digits with optional thousands separators,
# optional decimal part. e.g. "-1,234.5", "42", "3.".
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def gold_answer(solution):
    """Extract the reference number from a GSM8K solution.

    GSM8K solutions end with a line like '#### 1,234'; we return '1234'.
    """
    return solution.split("####")[-1].strip().replace(",", "")


def extract_answer(text):
    """Pull the model's final numeric answer out of its output (or None).

    Prefer the content of the <answer> tag; if the model ignored the format,
    fall back to the whole text. Either way take the *last* number, because
    in free-form reasoning the final answer usually comes last.
    """
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    s = m.group(1) if m else text
    nums = NUMBER_RE.findall(s)
    # Remove thousands separators and a sentence-final "." ("... is 42.").
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same_number(a, b):
    """Compare two numeric strings as floats ('42' == '42.0'); False if unparsable."""
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def is_well_formatted(text):
    """True if `text` follows the <think>...</think><answer>...</answer> structure."""
    return bool(FORMAT_RE.match(text))


# ---------------------------------------------------------------------------
# Reward functions in the signature TRL's GRPOTrainer expects:
#   f(prompts, completions, **dataset_columns) -> list[float], one per completion.
# With a conversational dataset each completion is a list of messages, and
# the generated text is completion[0]["content"].
# ---------------------------------------------------------------------------
def correctness_reward(prompts, completions, answer, **kw):
    """+1.0 for a completion whose final number equals the gold `answer`, else 0.0.

    `answer` is the dataset column of the same name; TRL passes extra columns
    as keyword arguments, aligned with `completions`.
    """
    return [1.0 if same_number(extract_answer(c[0]["content"]), a) else 0.0 for c, a in zip(completions, answer)]


def format_reward(prompts, completions, **kw):
    """+0.5 for a completion in the requested tag format, else 0.0.

    Kept smaller than the correctness reward so that "well formatted but
    wrong" is never worth more than "correct".
    """
    return [0.5 if is_well_formatted(c[0]["content"]) else 0.0 for c in completions]


def to_prompt(ex):
    """Map a raw GSM8K row to the {"prompt", "answer"} format used for training and evaluation."""
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ex["question"]}],
        "answer": gold_answer(ex["answer"]),
    }
