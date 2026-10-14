"""Tests for GSM8K parsing and the GRPO reward functions."""
import pytest

from common.reasoning import (
    SYSTEM_PROMPT,
    correctness_reward,
    extract_answer,
    format_reward,
    gold_answer,
    is_well_formatted,
    same_number,
    to_prompt,
)


def completion(text):
    """Wrap text the way TRL passes a conversational completion to a reward function."""
    return [{"role": "assistant", "content": text}]


def test_gold_answer_strips_marker_and_commas():
    assert gold_answer("She has 3 + 4 = 7 apples.\n#### 1,234") == "1234"


@pytest.mark.parametrize("text, expected", [
    ("<think>2+2</think><answer>4</answer>", "4"),
    ("<think>...</think><answer>The total is $1,250.</answer>", "1250"),
    ("<answer>-3.5</answer>", "-3.5"),
    ("no tags, first 3 then finally 12.", "12"),
    ("<think>I don't know</think><answer>unknown</answer>", None),
    ("", None),
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_extract_answer_prefers_answer_tag_over_reasoning():
    assert extract_answer("<think>maybe 10 or 20</think><answer>15</answer> then 99") == "15"


def test_same_number():
    assert same_number("42", "42.0")
    assert not same_number("41", "42")
    assert not same_number(None, "42")
    assert not same_number("abc", "42")


def test_is_well_formatted():
    assert is_well_formatted("<think>\nstep 1\nstep 2\n</think>\n<answer>7</answer>\n")
    assert not is_well_formatted("<answer>7</answer>")                       # no reasoning
    assert not is_well_formatted("<think>x</think><answer>7</answer> extra")  # trailing text
    assert not is_well_formatted("<think></think><answer>7</answer>")         # empty think


def test_correctness_reward():
    comps = [completion("<answer>7</answer>"), completion("<answer>8</answer>"), completion("seven")]
    assert correctness_reward(None, comps, answer=["7", "7", "7"]) == [1.0, 0.0, 0.0]


def test_format_reward():
    comps = [completion("<think>a</think><answer>1</answer>"), completion("1")]
    assert format_reward(None, comps) == [0.5, 0.0]


def test_correct_beats_formatted_but_wrong():
    right = completion("the answer is 7")
    pretty_wrong = completion("<think>a</think><answer>8</answer>")
    total = [c + f for c, f in zip(correctness_reward(None, [right, pretty_wrong], answer=["7", "7"]),
                                   format_reward(None, [right, pretty_wrong]))]
    assert total[0] > total[1]


def test_to_prompt():
    ex = to_prompt({"question": "What is 2+2?", "answer": "2+2=4\n#### 4"})
    assert ex["answer"] == "4"
    assert ex["prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert ex["prompt"][1] == {"role": "user", "content": "What is 2+2?"}
