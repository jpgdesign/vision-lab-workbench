import copy
import hashlib
import json

import pytest

from backend import workflow_samples as samples
from backend.workflow_templates import templates


def fixture_media(tmp_path, monkeypatch, sample_id="zoneguard", name="renamed-video.mp4"):
    data = ("verified sample bytes " + sample_id).encode()
    path = tmp_path / name
    path.write_bytes(data)
    specs = copy.deepcopy(samples.SAMPLES)
    specs[sample_id].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(samples, "SAMPLES", specs)
    return {"id": "local-clip", "name": name, "kind": specs[sample_id]["kind"], "path": str(path), "url": "https://not-the-media-route.example/"}


def template(document, ident):
    return next(value for value in document["templates"] if value["id"] == ident)


def test_all_published_templates_define_relevant_sample_guidance():
    document = samples.with_samples(templates(), [])
    assert set(samples.TEMPLATE_SAMPLES) == {t["id"] for t in document["templates"]}
    for value in document["templates"]:
        sample = value["sample"]
        assert sample["expected"] and sample["title"]
        assert sample["kind"] in {"image", "video"}
        assert sample["available"] is False and sample["media_id"] is None and sample["url"] is None
        expected_stride = 3 if value["id"].startswith("handwashing-") else 1
        assert value["recommended_options"]["frame_stride"] == expected_stride


def test_handwashing_defaults_cover_complete_22_second_clip():
    document = samples.with_samples(templates(), [])
    for ident in ("handwashing-hybrid", "handwashing-detection-mask"):
        options = template(document, ident)["recommended_options"]
        assert options["max_frames"] * options["frame_stride"] >= 660
        # COCO17 ST-GCN++ needs at least 32 observations in its 4 s window.
        assert 30 / options["frame_stride"] * 4 >= 32


def test_matching_uses_content_not_name_and_never_publishes_local_paths(tmp_path, monkeypatch):
    entry = fixture_media(tmp_path, monkeypatch)
    document = templates()
    before = copy.deepcopy(document)
    result = samples.with_samples(document, [entry])
    sample = template(result, "zone-guard")["sample"]
    assert sample["available"] and sample["media_id"] == "local-clip"
    assert sample["name"] == "renamed-video.mp4" and sample["origin"] == "original_project"
    assert sample["url"] == "/api/media/local-clip/content"
    assert "not-the-media-route" not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(result) and "\"path\"" not in json.dumps(result)
    assert document == before


def test_wrong_content_or_deleted_clip_is_not_misattributed(tmp_path, monkeypatch):
    entry = fixture_media(tmp_path, monkeypatch, name="0_te21_cropped.mp4")
    path = tmp_path / entry["name"]
    path.write_bytes(b"completely other file")
    assert not template(samples.with_samples(templates(), [entry]), "zone-guard")["sample"]["available"]
    path.unlink()
    assert not template(samples.with_samples(templates(), [entry]), "zone-guard")["sample"]["available"]


def test_known_content_with_wrong_media_kind_is_rejected(tmp_path, monkeypatch):
    entry = fixture_media(tmp_path, monkeypatch)
    entry["kind"] = "image"
    assert not template(samples.with_samples(templates(), [entry]), "zone-guard")["sample"]["available"]


def test_bundled_fallback_is_explicit_and_original_is_preferred(tmp_path, monkeypatch):
    zone = fixture_media(tmp_path, monkeypatch)
    traffic = fixture_media(tmp_path, monkeypatch, "traffic", "traffic.webm")
    traffic["id"] = "bundled-traffic"
    result = samples.with_samples(templates(), [traffic])
    chosen = template(result, "zone-guard")["sample"]
    assert chosen["origin"] == "bundled" and chosen["media_id"] == "bundled-traffic"
    assert "不是參考專案" in chosen["notes"]
    result = samples.with_samples(templates(), [traffic, zone])
    assert template(result, "zone-guard")["sample"]["origin"] == "original_project"


@pytest.mark.parametrize("ident", ["posture-fall", "ppe-safety"])
def test_safety_templates_do_not_offer_unrelated_available_traffic(tmp_path, monkeypatch, ident):
    entry = fixture_media(tmp_path, monkeypatch, "traffic")
    sample = template(samples.with_samples(templates(), [entry]), ident)["sample"]
    assert not sample["available"] and sample["origin"] == "user_required"
    assert "未提交" in sample["notes"]


def test_renamed_shared_sample_does_not_change_other_template_requirements(tmp_path, monkeypatch):
    entry = fixture_media(tmp_path, monkeypatch)
    result = samples.with_samples(templates(), [entry])
    assert template(result, "opencv-roi-monitor")["sample"]["available"]
    assert not template(result, "skeleton-actions")["sample"]["available"]
    assert not template(result, "rgb-action-rules")["sample"]["available"]


def test_malformed_registry_rows_do_not_crash_template_list():
    result = samples.with_samples(templates(), [None, {}, {"id": "x", "path": None}, {"path": "missing", "id": []}])
    assert all(not t["sample"]["available"] for t in result["templates"])
