"""
Function-calling (tool-use) helpers for `agents/function-calling-llm`.

The model must either emit exactly one tool call

    <functioncall>{"name": "get_exchange_rate", "arguments": {"base": "USD", "target": "INR"}}</functioncall>

or answer in plain text when no tool applies. This module contains:

* `parse_glaive_example` - converts glaive-function-calling-v2's raw text
  format into (system, user, target, call).
* `fc_messages`          - the chat prompt (identical before/after training).
* `parse_call`           - parses a model output back into a JSON call.
* `call_metrics`         - the four agent metrics reported in the README.
"""

import json
import re

# Appended to every system prompt: tells the model the exact output contract.
INSTRUCTION = (
    'If one of the functions is needed, reply ONLY with <functioncall>{"name": ..., "arguments": {...}}'
    "</functioncall>. Otherwise answer the user directly."
)

# Grab the JSON object after <functioncall>. The closing tag is optional
# (generation may stop right after the "}"), and "\{.*\}" is greedy so nested
# braces inside "arguments" are kept. re.S: the JSON may span lines.
CALL_RE = re.compile(r"<functioncall>\s*(\{.*\})\s*(?:</functioncall>|$)", re.S)

# Returned for rows we cannot use; `ok=False` lets `datasets.filter` drop them.
_EMPTY = {"ok": False, "system": "", "user": "", "target": "", "call": ""}


def parse_glaive_example(ex):
    """Turn one raw glaive row into {"ok", "system", "user", "target", "call"}.

    glaive stores a conversation as one string:
        USER: ... ASSISTANT: <functioncall> {"name": "f", "arguments": '{"x": 1}'} <|endoftext|> FUNCTION RESPONSE: ...
    We keep only the *first* user turn and the assistant's *first* reply.

    Note the quirk: glaive writes "arguments" as a single-quoted JSON *string*.
    We decode it and re-serialise the whole call as clean JSON, so the target
    the model learns (and the gold we score against) is always valid JSON.
    """
    system = ex["system"].replace("SYSTEM:", "").strip()
    chat = ex["chat"]
    # First USER: ... ASSISTANT: ... pair; the reply ends at end-of-text, at a
    # function response, at the next user turn, or at the end of the string.
    m = re.search(r"USER:(.*?)ASSISTANT:(.*?)(?:<\|endoftext\|>|FUNCTION RESPONSE:|USER:|$)", chat, re.S)
    if not m:
        return dict(_EMPTY)
    user, reply = m.group(1).strip(), m.group(2).strip()
    call = ""
    if "<functioncall>" in reply:
        # name: the quoted string after "name":   arguments: everything between the single quotes.
        fm = re.search(r'"name":\s*"([^"]+)".*?"arguments":\s*\'(.*)\'\s*\}', reply, re.S)
        try:
            call = json.dumps({"name": fm.group(1), "arguments": json.loads(fm.group(2))}, ensure_ascii=False)
        except Exception:
            # Unparsable call in the source data (or fm is None) - skip the row
            # rather than train the model on broken JSON.
            return dict(_EMPTY)
        target = f"<functioncall>{call}</functioncall>"
    else:
        target = reply  # no tool needed: the target is the plain-text answer
    return {"ok": bool(user and target), "system": system, "user": user, "target": target, "call": call}


def fc_messages(ex, with_answer):
    """Chat messages for one example: system (tool schemas + INSTRUCTION), user, and optionally the target."""
    msgs = [{"role": "system", "content": f"{ex['system']}\n\n{INSTRUCTION}"}, {"role": "user", "content": ex["user"]}]
    if with_answer:
        msgs.append({"role": "assistant", "content": ex["target"]})
    return msgs


def parse_call(text):
    """Parse a model output into (call_dict_or_None, is_valid_call).

    A call is *valid* when the text after <functioncall> is JSON describing an
    object with a "name" key.
    """
    m = CALL_RE.search(text)
    if not m:
        return None, False
    try:
        c = json.loads(m.group(1))
        return c, isinstance(c, dict) and "name" in c
    except Exception:
        return None, False


def call_metrics(outputs, gold_calls):
    """Score generated texts against gold calls.

    outputs    : list of generated texts.
    gold_calls : list aligned with `outputs`; each is the gold call as a JSON
                 string, or "" when the correct behaviour is NOT to call a tool.

    Returns
        call_decision_accuracy - called a tool exactly when it should (over all examples)
        tool_name_accuracy     - right function name   (over examples that need a call)
        full_call_exact_match  - right name AND arguments equal as parsed JSON (same denominator)
        valid_json_rate        - parsable call         (over examples where a call was attempted)
        test_examples, examples_needing_a_call - the denominators
    """
    n_call = decision = name_ok = exact = attempted = valid = 0
    for text, gold_call in zip(outputs, gold_calls):
        pred, ok_json = parse_call(text)
        should = bool(gold_call)
        called = "<functioncall>" in text
        decision += called == should
        if called:
            attempted += 1
            valid += ok_json
        if should:
            n_call += 1
            gold = json.loads(gold_call)
            if ok_json and pred.get("name") == gold["name"]:
                name_ok += 1
                # Comparing parsed dicts ignores key order and whitespace,
                # but values must match exactly ("5" != 5).
                exact += pred.get("arguments") == gold["arguments"]
    n = len(outputs)
    # max(..., 1) avoids a division by zero on tiny smoke-test sets.
    return {
        "call_decision_accuracy": decision / n,
        "tool_name_accuracy": name_ok / max(n_call, 1),
        "full_call_exact_match": exact / max(n_call, 1),
        "valid_json_rate": valid / max(attempted, 1),
        "test_examples": n,
        "examples_needing_a_call": n_call,
    }
