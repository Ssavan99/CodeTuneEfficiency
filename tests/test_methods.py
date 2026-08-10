"""Every method must train exactly what it claims to train.

These build a tiny randomly-initialised RoBERTa locally, so the suite runs
offline in seconds and never downloads a 500 MB checkpoint.
"""

import pytest
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification

from codetune.cost import count_parameters, delta_checkpoint_bytes, trainable_state_dict
from codetune.methods import METHODS, apply_method


def tiny_model():
    config = RobertaConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=2,
    )
    torch.manual_seed(0)
    return RobertaForSequenceClassification(config)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_something_is_trainable(method):
    model = apply_method(tiny_model(), method)
    assert count_parameters(model)["trainable_params"] > 0


@pytest.mark.parametrize("method", sorted(METHODS))
def test_classifier_head_always_trains(method):
    """The head is randomly initialised — freezing it makes learning impossible."""
    model = apply_method(tiny_model(), method)
    head = [p for n, p in model.named_parameters() if "classifier" in n or "score" in n]
    assert head, f"no classifier parameters found for {method}"
    assert any(p.requires_grad for p in head)


def test_full_trains_everything():
    counts = count_parameters(apply_method(tiny_model(), "full"))
    assert counts["trainable_params"] == counts["total_params"]
    assert counts["trainable_pct"] == pytest.approx(100.0)


def test_bitfit_trains_only_biases_and_head():
    model = apply_method(tiny_model(), "bitfit")
    for name, param in model.named_parameters():
        is_head = "classifier" in name or "score" in name
        if param.requires_grad:
            assert name.endswith(".bias") or is_head, f"{name} should be frozen under bitfit"
        else:
            assert not name.endswith(".bias") or is_head is False


def test_peft_methods_are_far_cheaper_than_full():
    full = count_parameters(apply_method(tiny_model(), "full"))["trainable_params"]
    for method in ("bitfit", "lora", "parallel_adapter"):
        cheap = count_parameters(apply_method(tiny_model(), method))["trainable_params"]
        assert cheap < full, f"{method} trains no fewer parameters than full fine-tuning"


def test_parallel_adapter_is_identity_at_init():
    """Zero-initialised up-projection means the pre-trained function is untouched at step 0."""
    baseline = tiny_model().eval()
    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        before = baseline(input_ids=ids).logits

    adapted = apply_method(baseline, "parallel_adapter").eval()
    with torch.no_grad():
        after = adapted(input_ids=ids).logits

    assert torch.allclose(before, after, atol=1e-6)


def test_parallel_adapter_adds_one_adapter_per_layer():
    model = apply_method(tiny_model(), "parallel_adapter")
    adapters = {n.split(".adapter.")[0] for n, _ in model.named_parameters() if ".adapter." in n}
    assert len(adapters) == model.config.num_hidden_layers


def test_delta_checkpoint_matches_trainable_parameters():
    model = apply_method(tiny_model(), "bitfit")
    state = trainable_state_dict(model)
    assert state, "trainable state dict should not be empty"
    expected = sum(t.numel() * t.element_size() for t in state.values())
    assert delta_checkpoint_bytes(model) == expected
    assert delta_checkpoint_bytes(model) < delta_checkpoint_bytes(apply_method(tiny_model(), "full"))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        apply_method(tiny_model(), "definitely_not_a_method")


def test_adapted_model_still_produces_gradients():
    model = apply_method(tiny_model(), "parallel_adapter")
    out = model(input_ids=torch.randint(0, 128, (2, 16)), labels=torch.tensor([0, 1]))
    out.loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
