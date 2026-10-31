"""Long-caption preservation and CPU model save/reload without downloaded weights."""
from types import SimpleNamespace

import pytest
import torch
from transformers import BertConfig, GroundingDinoConfig, GroundingDinoForObjectDetection, SwinConfig
from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps

from backend.training_grounding import prepare_training_caption, _supervised_loss


class WhitespaceTokenizer:
    """Small deterministic test tokenizer with the native BERT delimiters."""
    def __call__(self, caption, *, return_tensors, truncation):
        assert return_tensors == "pt" and truncation is False
        tokens = [101] + [1012 if word == "." else 200 + index % 500
                          for index, word in enumerate(caption.split())] + [102]
        ids = torch.tensor([tokens], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "token_type_ids": torch.zeros_like(ids)}


def config(maximum=256, positions=512):
    return SimpleNamespace(max_text_len=maximum,
                           text_config=SimpleNamespace(max_position_embeddings=positions))


def test_short_caption_preserves_existing_capacity_and_exact_phrase_order():
    settings = config()
    caption, count, limits = prepare_training_caption(["Traffic Light", "dog"], WhitespaceTokenizer(), settings)
    assert caption == "traffic light . dog ."
    assert count == 7 and settings.max_text_len == 256
    assert limits == {"base_max_text_len": 256, "max_text_len": 256, "text_backbone_max_position_embeddings": 512}


def test_long_caption_expands_only_to_required_length_without_losing_last_class():
    settings = config()
    classes = [f"food item {index}" for index in range(91)]
    caption, count, limits = prepare_training_caption(classes, WhitespaceTokenizer(), settings)
    assert count == 366 and settings.max_text_len == 366
    assert caption == " . ".join(classes) + " ."
    assert limits["base_max_text_len"] == 256
    encoded = WhitespaceTokenizer()(caption, return_tensors="pt", truncation=False)
    mapping = build_label_maps(torch.empty((1, 2, settings.max_text_len)), encoded["input_ids"])[0]
    assert mapping.shape == (91, 366)
    assert mapping[-1, 256:].sum() == 3
    assert (mapping.sum(-1) == 3).all()


def test_caption_exactly_at_position_capacity_is_accepted():
    settings = config()
    caption, count, _ = prepare_training_caption(["food " * 509], WhitespaceTokenizer(), settings)
    assert count == settings.max_text_len == 512
    assert caption.count("food") == 509


def test_over_position_capacity_fails_before_mutating_config():
    settings = config()
    with pytest.raises(ValueError, match="513.*512"):
        prepare_training_caption(["food " * 510], WhitespaceTokenizer(), settings)
    assert settings.max_text_len == 256


@pytest.mark.parametrize("classes", [["Dog", "dog"], ["food. rice"], ["[MASK]"]])
def test_expansion_does_not_relax_class_identity_validation(classes):
    settings = config()
    with pytest.raises(ValueError):
        prepare_training_caption(classes, WhitespaceTokenizer(), settings)
    assert settings.max_text_len == 256


@pytest.mark.parametrize("maximum,positions", [(True, 512), (256, None), (0, 512), (256, 512.0), (513, 512)])
def test_invalid_or_inconsistent_local_position_limits_are_rejected(maximum, positions):
    with pytest.raises(ValueError):
        prepare_training_caption(["food"], WhitespaceTokenizer(), config(maximum, positions))


def tiny_config():
    return GroundingDinoConfig(
        backbone_config=SwinConfig(image_size=64, embed_dim=8, depths=[1, 1, 1, 1],
            num_heads=[1, 1, 2, 4], window_size=2, out_features=["stage2", "stage3", "stage4"]),
        text_config=BertConfig(vocab_size=1100, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=4, intermediate_size=64, max_position_embeddings=512),
        d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=4,
        decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64,
        encoder_n_points=2, decoder_n_points=2, num_feature_levels=3, num_queries=3,
        two_stage=True, auxiliary_loss=False, disable_custom_kernels=True,
        dropout=0.0, attention_dropout=0.0, max_text_len=256,
    )


def test_cpu_full_tiny_model_long_caption_backward_and_checkpoint_reload(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        settings = tiny_config()
        # 443 tokens matches the complete 91-class BentoCal caption counted
        # with the local pretrained tokenizer; no dataset is needed in CI.
        classes = [f"food item {index}" + (" extra" if index < 77 else "") for index in range(91)]
        caption, count, _ = prepare_training_caption(classes, WhitespaceTokenizer(), settings)
        assert count == 443
        inputs = WhitespaceTokenizer()(caption, return_tensors="pt", truncation=False)
        inputs.update(pixel_values=torch.randn(1, 3, 64, 64),
                      pixel_mask=torch.ones(1, 64, 64, dtype=torch.long))
        model = GroundingDinoForObjectDetection(settings).cpu().eval()
        assert all(head.max_text_len == count for head in model.class_embed)
        with torch.no_grad():
            before = model(**inputs).logits
        assert before.shape == (1, 3, count)
        supervised = {**inputs, "labels": [{"class_labels": torch.tensor([90]),
                        "boxes": torch.tensor([[.5, .5, .2, .2]])}]}
        loss, _ = _supervised_loss(model, supervised)
        assert torch.isfinite(loss)
        loss.backward()
        # Last class lies beyond the old 256-token cutoff. It remains in the
        # native matching/loss and gradients reach both text and box pathways.
        assert model.model.text_projection.weight.grad.abs().sum() > 0
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for name, parameter in model.named_parameters() if "bbox_embed" in name)
        model.save_pretrained(tmp_path, safe_serialization=True)
        restored, loading = GroundingDinoForObjectDetection.from_pretrained(
            tmp_path, local_files_only=True, output_loading_info=True)
        assert not loading["missing_keys"] and not loading["unexpected_keys"] and not loading["mismatched_keys"]
        assert restored.config.max_text_len == count
        assert restored.config.text_config.max_position_embeddings == 512
        assert all(head.max_text_len == count for head in restored.class_embed)
        restored.eval()
        with torch.no_grad():
            after = restored(**inputs).logits
        torch.testing.assert_close(after, before)
    finally:
        torch.set_num_threads(previous_threads)
