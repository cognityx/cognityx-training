from __future__ import annotations

from types import SimpleNamespace

import torch

from cognityx_training.custom_pytorch import _evaluate_model


class BatchEncodingTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, token_ids, **kwargs):
        assert token_ids.tolist() == [9, 8]
        return "Expected answer"


class FakeModel:
    def __init__(self) -> None:
        self.training = True
        self.generated_with = None

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def generate(self, **kwargs):
        self.generated_with = kwargs
        assert kwargs["input_ids"].shape[-1] == 3
        assert kwargs["attention_mask"].shape[-1] == 3
        return torch.tensor([[1, 2, 3, 9, 8]])


def test_evaluation_accepts_batch_encoding_and_reports_suite_metrics():
    tokenizer = BatchEncodingTokenizer()
    model = FakeModel()
    record = SimpleNamespace(
        record_id="eval-1",
        messages=(
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Expected answer"},
        ),
        metadata={
            "research_role": "paraphrase_evaluation",
            "evaluation_set_id": "paraphrase-1",
            "evidence_ids": ["evidence-1"],
        },
    )
    result = _evaluate_model(model, tokenizer, [record], torch.device("cpu"), 8, torch)
    assert tokenizer.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    assert result["exact_match_accuracy"] == 1.0
    assert result["suite_metrics"]["paraphrase_evaluation"]["example_count"] == 1
    assert model.training is True
