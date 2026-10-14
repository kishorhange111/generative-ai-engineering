"""Tests for the glaive parser, the tool-call parser and the agent metrics."""

import json

from common.function_calling import INSTRUCTION, call_metrics, fc_messages, parse_call, parse_glaive_example

SYSTEM = 'SYSTEM: You are a helpful assistant with access to the following functions. {"name": "get_rate"}'


def test_parse_glaive_function_call():
    chat = (
        'USER: Convert 100 USD to INR. ASSISTANT: <functioncall> {"name": "get_rate", '
        '"arguments": \'{"base": "USD", "target": "INR"}\'} <|endoftext|> FUNCTION RESPONSE: {"rate": 83}'
    )
    ex = parse_glaive_example({"system": SYSTEM, "chat": chat})
    assert ex["ok"]
    assert ex["user"] == "Convert 100 USD to INR."
    assert json.loads(ex["call"]) == {"name": "get_rate", "arguments": {"base": "USD", "target": "INR"}}
    assert ex["target"] == f"<functioncall>{ex['call']}</functioncall>"
    assert not ex["system"].startswith("SYSTEM:")


def test_parse_glaive_plain_answer():
    chat = "USER: Tell me a joke. ASSISTANT: I can't tell jokes with my tools, but here is one... <|endoftext|>"
    ex = parse_glaive_example({"system": SYSTEM, "chat": chat})
    assert ex["ok"] and ex["call"] == ""
    assert ex["target"].startswith("I can't tell jokes")


def test_parse_glaive_rejects_broken_rows():
    assert not parse_glaive_example({"system": SYSTEM, "chat": "no speaker tags"})["ok"]
    broken = 'USER: hi ASSISTANT: <functioncall> {"name": "f", "arguments": \'{bad json}\'} <|endoftext|>'
    assert not parse_glaive_example({"system": SYSTEM, "chat": broken})["ok"]


def test_fc_messages():
    ex = {"system": "tools", "user": "hi", "target": "hello"}
    msgs = fc_messages(ex, False)
    assert msgs[0]["content"] == f"tools\n\n{INSTRUCTION}"
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert fc_messages(ex, True)[-1] == {"role": "assistant", "content": "hello"}


def test_parse_call():
    call, ok = parse_call('<functioncall>{"name": "f", "arguments": {"x": {"y": 1}}}</functioncall>')
    assert ok and call == {"name": "f", "arguments": {"x": {"y": 1}}}
    # closing tag missing (generation stopped) is still accepted
    assert parse_call('<functioncall> {"name": "f", "arguments": {}}')[1]
    assert parse_call("plain answer") == (None, False)
    assert parse_call("<functioncall>{not json}</functioncall>") == (None, False)
    assert parse_call('<functioncall>{"arguments": {}}</functioncall>')[1] is False  # no name


def test_call_metrics():
    gold = json.dumps({"name": "f", "arguments": {"x": 1}})
    outputs = [
        '<functioncall>{"name": "f", "arguments": {"x": 1}}</functioncall>',  # perfect
        '<functioncall>{"name": "f", "arguments": {"x": 2}}</functioncall>',  # right tool, wrong args
        "Sure! Here you go.",  # should have called
        "Hello there",  # correctly no call
        "<functioncall>{oops}</functioncall>",  # called when it should not
    ]
    m = call_metrics(outputs, [gold, gold, gold, "", ""])
    assert m["test_examples"] == 5 and m["examples_needing_a_call"] == 3
    assert m["call_decision_accuracy"] == 3 / 5
    assert m["tool_name_accuracy"] == 2 / 3
    assert m["full_call_exact_match"] == 1 / 3
    assert m["valid_json_rate"] == 2 / 3


def test_call_metrics_argument_key_order_does_not_matter():
    gold = json.dumps({"name": "f", "arguments": {"a": 1, "b": 2}})
    m = call_metrics(['<functioncall>{"name": "f", "arguments": {"b": 2, "a": 1}}</functioncall>'], [gold])
    assert m["full_call_exact_match"] == 1.0
