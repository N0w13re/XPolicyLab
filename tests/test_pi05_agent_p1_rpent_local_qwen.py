import json

from XPolicyLab.policy.Pi_05_Agent_P1_RPent.local_qwen_server import (
    _openai_message,
)


def test_a_well_formed_tool_call_becomes_an_openai_tool_call():
    text = (
        "<tool_call>"
        '{"name": "sample_world_xyz", "arguments": {"view": "head", '
        '"pixels": [[780, 440]], "step": 0}}'
        "</tool_call>"
    )

    message = _openai_message(text)

    call = message["tool_calls"][0]["function"]
    assert call["name"] == "sample_world_xyz"
    assert json.loads(call["arguments"])["pixels"] == [[780, 440]]
    assert message["content"] == ""


def test_a_malformed_tool_call_is_reported_as_text_instead_of_raising():
    text = (
        "<tool_call>"
        '{"name": "sample_world_xyz", "arguments": {"view": "head", '
        '"pixels": [[780, 440]] "step": 0}}'
        "</tool_call>"
    )

    message = _openai_message(text)

    assert "tool_calls" not in message
    assert "Discarded unparsable tool_call blocks:" in message["content"]


def test_a_nameless_tool_call_is_reported_as_text():
    message = _openai_message('<tool_call>{"arguments": {}}</tool_call>')

    assert "tool_calls" not in message
    assert "needs a name field" in message["content"]


def test_one_broken_call_does_not_discard_a_valid_sibling():
    text = (
        '<tool_call>{"name": "broken", "arguments": {,}}</tool_call>'
        '<tool_call>{"name": "finish", "arguments": {"status": "failure", '
        '"summary": "stop"}}</tool_call>'
    )

    message = _openai_message(text)

    assert [call["function"]["name"] for call in message["tool_calls"]] == [
        "finish"
    ]
