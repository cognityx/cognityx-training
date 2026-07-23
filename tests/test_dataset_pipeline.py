import json

import pytest

from cognityx_training.dataset_pipeline import (
    load_jsonl_records,
    messages_for_record,
    partition_records,
    render_ift_examples,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return "|".join(f"{item['role']}:{item['content']}" for item in messages)


def test_load_and_render_instruction_dataset(tmp_path) -> None:
    path = tmp_path / "hello.jsonl"
    path.write_text(
        json.dumps({"instruction": "Say hello", "output": "Hello!"}) + "\n",
        encoding="utf-8",
    )

    records = load_jsonl_records(str(path))
    rendered = render_ift_examples(records, FakeTokenizer())

    assert rendered == ["user:Say hello|assistant:Hello!"]


def test_messages_are_preserved() -> None:
    record = {"messages": [{"role": "user", "content": "Hi"}]}

    assert messages_for_record(record) == record["messages"]


def test_unsupported_dataset_provider_is_explicit() -> None:
    with pytest.raises(ValueError, match="local files only"):
        load_jsonl_records("s3://bucket/train.jsonl")


def test_partition_caps_training_and_preserves_held_out_evaluation() -> None:
    records = [
        {"instruction": "train 1", "output": "one"},
        {"instruction": "train 2", "output": "two"},
        {"split": "evaluation", "instruction": "evaluate", "output": "answer"},
    ]

    training, evaluation = partition_records(records, max_examples=1)

    assert training == records[:1]
    assert evaluation == records[2:]
