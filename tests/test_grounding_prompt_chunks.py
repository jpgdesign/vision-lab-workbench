"""Complete vocabularies, real native negative-only loss, and accumulation math."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import (BatchEncoding, BertConfig, GroundingDinoConfig,
                          GroundingDinoForObjectDetection, SwinConfig)

from backend.detection_expansion import grounding_predictor
from backend.grounding_prompts import (build_caption_chunks, chunk_loss_weight,
                                      prepare_caption_chunks, remap_chunk_labels,
                                      validate_caption_chunks)
from backend.training_grounding import _GroupedCollator, _supervised_loss


class Tokenizer:
    def __call__(self, caption, *, return_tensors, truncation, verbose=True):
        assert truncation is False
        ids = [101] + [1012 if word == "." else 200 + index % 500
                       for index, word in enumerate(caption.split())] + [102]
        values = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": values, "attention_mask": torch.ones_like(values),
                "token_type_ids": torch.zeros_like(values)}


def test_full_170_class_vocabulary_keeps_every_global_id_and_exact_label():
    classes = [f"food item {i}" for i in range(170)]
    config = SimpleNamespace(max_text_len=256, text_config=SimpleNamespace(max_position_embeddings=512))
    chunks, limits = prepare_caption_chunks(classes, Tokenizer(), config)
    assert len(chunks) == 2
    assert [i for chunk in chunks for i in chunk["global_class_ids"]] == list(range(170))
    assert all(chunk["text_tokens"] <= 512 for chunk in chunks)
    assert chunks[-1]["caption"].endswith("food item 169 .")
    assert config.max_text_len == max(chunk["text_tokens"] for chunk in chunks)
    assert limits["text_backbone_max_position_embeddings"] == 512
    assert validate_caption_chunks(chunks, classes, Tokenizer(), config.max_text_len) is chunks
    malformed = deepcopy(chunks)
    malformed[-1]["global_class_ids"][-1] = 0
    with pytest.raises(ValueError):
        validate_caption_chunks(malformed, classes, Tokenizer(), 512)
    with pytest.raises(ValueError, match="完整"):
        validate_caption_chunks(chunks[:-1], classes, Tokenizer(), 512)


def test_small_caption_keeps_identical_text_and_capacity_and_bad_identity_is_not_hidden_by_chunks():
    config = SimpleNamespace(max_text_len=256, text_config=SimpleNamespace(max_position_embeddings=512))
    chunks, _ = prepare_caption_chunks(["Traffic Light", "dog"], Tokenizer(), config)
    assert chunks == [{"caption": "traffic light . dog .", "global_class_ids": [0, 1], "text_tokens": 7}]
    assert config.max_text_len == 256
    with pytest.raises(ValueError, match="重複"):
        build_caption_chunks(["Dog", "rice", "dog"], Tokenizer(), 5)
    with pytest.raises(ValueError, match="上限"):
        build_caption_chunks(["food " * 510], Tokenizer(), 512)


def test_saved_chunks_cannot_hide_duplicate_global_class_names():
    chunks = [{"caption": "dog .", "global_class_ids": [0], "text_tokens": 4},
              {"caption": "dog .", "global_class_ids": [1], "text_tokens": 4}]
    with pytest.raises(ValueError, match="重複"):
        validate_caption_chunks(chunks, ["Dog", " dog "], Tokenizer(), 512)


def test_remapping_keeps_every_box_once_and_supports_negative_only_caption():
    target = {"class_labels": torch.tensor([0, 169, 169]),
              "boxes": torch.tensor([[.1, .1, .1, .1], [.5, .5, .2, .2], [.8, .8, .1, .1]])}
    chunks = [[0, 5], [100, 169], [3, 4]]
    outputs = [remap_chunk_labels([target], ids)[0] for ids in chunks]
    assert [out["class_labels"].tolist() for out in outputs] == [[0], [1, 1], []]
    assert outputs[-1]["boxes"].shape == (0, 4)
    recovered = [(ids[int(local)], box.tolist()) for ids, out in zip(chunks, outputs)
                 for local, box in zip(out["class_labels"], out["boxes"])]
    assert recovered == [(int(label), box.tolist()) for label, box in zip(target["class_labels"], target["boxes"])]
    assert target["class_labels"].tolist() == [0, 169, 169]


def test_grouped_collator_uses_same_augmented_images_and_never_skips_empty_targets():
    observed = []
    class Processor:
        def __call__(self, images, text, **kwargs):
            observed.append((images, text))
            return {"pixel_values": torch.zeros(len(images), 3, 8, 8)}
    chunks = [{"caption": "old .", "global_class_ids": [0]}, {"caption": "new .", "global_class_ids": [169]}]
    image = Image.new("RGB", (8, 8))
    labels = {"class_labels": torch.tensor([169]), "boxes": torch.tensor([[.5, .5, .2, .2]])}
    batches = _GroupedCollator(Processor(), chunks)([(image, labels)])
    assert len(batches) == 2 and all(call[0][0] is image for call in observed)
    assert batches[0]["labels"][0]["class_labels"].numel() == 0
    assert batches[1]["labels"][0]["class_labels"].tolist() == [0]


@pytest.mark.parametrize("batch_sizes", [[2, 1], [1]])
def test_chunk_and_accumulation_weights_equal_one_mean_objective_even_for_short_last_window(batch_sizes):
    values = torch.tensor([[.1, .7], [.5, .9], [.2, .3]])[:sum(batch_sizes)]
    parameter = torch.tensor(.4, requires_grad=True)
    start = 0
    for count in batch_sizes:
        for chunk in range(2):
            loss = (parameter - values[start:start + count, chunk]).square().mean()
            (loss * chunk_loss_weight(count, len(values), 2)).backward()
        start += count
    reference = torch.tensor(.4, requires_grad=True)
    (reference - values).square().mean().backward()
    torch.testing.assert_close(parameter.grad, reference.grad)


def test_predictor_visits_all_chunks_and_preserves_global_ids_without_cross_chunk_score_normalization():
    class Model:
        config = SimpleNamespace(max_text_len=6)
        def eval(self): return self
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[[-10., 2., -10., -10.]]]),
                                   pred_boxes=torch.tensor([[[.5, .5, .4, .4]]]))
    seen = []
    class Processor:
        def __call__(self, **kwargs):
            seen.append(kwargs["text"])
            return BatchEncoding({"input_ids": torch.tensor([[101, 1000, 1012, 102]])})
    chunks = [{"caption": "old .", "global_class_ids": [0]}, {"caption": "new .", "global_class_ids": [169]}]
    predict = grounding_predictor(Model(), Processor(), "", {"imgsz": 64, "confidence": .1}, "cpu", caption_chunks=chunks)
    results = predict(Image.new("RGB", (100, 50)))
    assert seen == ["old .", "new ."]
    assert [row["class_id"] for row in results] == [0, 169]
    assert all(row["score"] == pytest.approx(torch.sigmoid(torch.tensor(2.)).item()) for row in results)
    assert results[0]["box"] == pytest.approx([30, 15, 70, 35])


def test_native_two_stage_model_negative_only_and_positive_chunks_backward_save_reload(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = GroundingDinoConfig(
            backbone_config=SwinConfig(image_size=64, embed_dim=8, depths=[1, 1, 1, 1],
                num_heads=[1, 1, 2, 4], window_size=2, out_features=["stage2", "stage3", "stage4"]),
            text_config=BertConfig(vocab_size=1100, hidden_size=32, num_hidden_layers=1,
                num_attention_heads=4, intermediate_size=64, max_position_embeddings=512),
            d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=4,
            decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64,
            encoder_n_points=2, decoder_n_points=2, num_feature_levels=3, num_queries=3,
            two_stage=True, auxiliary_loss=False, disable_custom_kernels=True,
            dropout=0.0, attention_dropout=0.0, max_text_len=256)
        classes = [f"food item {i}" for i in range(170)]
        chunks, _ = prepare_caption_chunks(classes, Tokenizer(), config)
        model = GroundingDinoForObjectDetection(config).cpu().eval()
        model.model.backbone.requires_grad_(False)
        model.model.text_backbone.requires_grad_(False)
        pixels = torch.randn(1, 3, 64, 64)
        targets = [{"class_labels": torch.tensor([169]), "boxes": torch.tensor([[.5, .5, .2, .2]])}]
        losses = []
        for chunk in chunks:
            inputs = Tokenizer()(chunk["caption"], return_tensors="pt", truncation=False)
            inputs.update(pixel_values=pixels, pixel_mask=torch.ones(1, 64, 64, dtype=torch.long),
                          labels=remap_chunk_labels(targets, chunk["global_class_ids"]))
            loss, components = _supervised_loss(model, inputs)
            assert torch.isfinite(loss)
            losses.append(float(loss.detach()))
            (loss / len(chunks)).backward()
        assert len(losses) == 2 and all(value > 0 for value in losses)
        assert model.model.text_projection.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.model.backbone.parameters())
        assert all(p.grad is None for p in model.model.text_backbone.parameters())
        model.save_pretrained(tmp_path, safe_serialization=True)
        restored, loading = GroundingDinoForObjectDetection.from_pretrained(tmp_path, local_files_only=True, output_loading_info=True)
        assert not loading["missing_keys"] and not loading["mismatched_keys"]
        assert restored.config.max_text_len == config.max_text_len
        assert validate_caption_chunks(chunks, classes, Tokenizer(), restored.config.max_text_len) is chunks
    finally:
        torch.set_num_threads(old_threads)
