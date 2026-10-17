import importlib
import json

import pytest

from backend import catalog, model_licenses
from backend.training_rfdetr import normalize_training_params, training_capabilities


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # Never consume the user's genuine saved license acknowledgement.
    root = tmp_path / "isolated-runtime"
    root.mkdir()
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(root))
    weight = tmp_path / "test-weight.pt"
    weight.write_bytes(b"fixture")
    monkeypatch.setattr(catalog, "WEIGHTS", {key: weight for key in catalog.WEIGHTS})
    monkeypatch.setattr(catalog.importlib.util, "find_spec", lambda name: object())
    return root


def consent():
    return {"schema_version": 1, "models": {"rf_detr_2xl": {
        "license_id": "PML-1.0",
        "license_url": "https://github.com/roboflow/rf-detr_plus/blob/main/LICENSE",
        "accepted": True, "accepted_at": "2026-10-18T03:00:00+08:00", "source": "user_confirmation",
    }}}


def save(runtime, value):
    (runtime / "model_license_acceptance.json").write_text(json.dumps(value), encoding="utf-8")


def rf_model():
    return next(model for model in catalog.get_models() if model["id"] == "rf_detr_2xl")


def test_persisted_specific_license_survives_module_reload_and_runtime_switch(runtime, monkeypatch):
    assert not model_licenses.is_model_license_accepted("rf_detr_2xl")
    save(runtime, consent())
    assert model_licenses.is_model_license_accepted("rf_detr_2xl")
    importlib.reload(model_licenses)
    assert model_licenses.is_model_license_accepted("rf_detr_2xl")
    assert not model_licenses.is_model_license_accepted("yolo26x")
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(runtime / "different-workspace"))
    assert not model_licenses.is_model_license_accepted("rf_detr_2xl")
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(runtime))
    assert model_licenses.is_model_license_accepted("rf_detr_2xl")


def test_catalog_and_training_defaults_use_saved_acceptance(runtime):
    save(runtime, consent())
    model = rf_model()
    assert model["license_accepted"] is True
    assert model["license_url"] == model_licenses.RF_DETR_LICENSE_URL
    assert next(p for p in model["capabilities"]["inference_params"] if p["key"] == "accept_platform_license")["default"] is True
    assert model["capabilities"]["training"]["defaults"]["accept_platform_license"] is True
    assert training_capabilities()["defaults"]["accept_platform_license"] is True


@pytest.mark.parametrize("params", [{}, {"accept_platform_license": False}, {"accept_platform_license": True}])
def test_saved_acceptance_normalizes_missing_or_stale_checkbox_for_execution(runtime, params):
    save(runtime, consent())
    assert catalog.validate_job_request("rf_detr_2xl", "inference", params)["accept_platform_license"] is True
    assert catalog.validate_job_request("rf_detr_2xl", "training", params)["accept_platform_license"] is True
    assert normalize_training_params(params)["accept_platform_license"] is True


@pytest.mark.parametrize("invalid", ["true", "false", 0, 1, None, []])
def test_saved_consent_does_not_bypass_boolean_type_validation(runtime, invalid):
    save(runtime, consent())
    with pytest.raises(ValueError, match="布林"):
        catalog.validate_job_request("rf_detr_2xl", "inference", {"accept_platform_license": invalid})
    with pytest.raises(ValueError, match="true / false"):
        normalize_training_params({"accept_platform_license": invalid})


def test_without_saved_consent_existing_explicit_gate_is_preserved(runtime):
    assert rf_model()["license_accepted"] is False
    assert training_capabilities()["defaults"]["accept_platform_license"] is False
    for params in ({}, {"accept_platform_license": False}):
        with pytest.raises(ValueError, match="授權"):
            catalog.validate_job_request("rf_detr_2xl", "inference", params)
        with pytest.raises(ValueError, match="License"):
            normalize_training_params(params)
    assert catalog.validate_job_request("rf_detr_2xl", "inference", {"accept_platform_license": True})["accept_platform_license"] is True
    assert normalize_training_params({"accept_platform_license": True})["accept_platform_license"] is True
    assert not (runtime / "model_license_acceptance.json").exists()


@pytest.mark.parametrize("field,value", [("accepted", False), ("accepted", "true"), ("accepted", 1),
                                         ("license_id", "PML-2.0"), ("license_url", "https://example.com/license")])
def test_other_versions_or_nonboolean_acknowledgements_do_not_grant_consent(runtime, field, value):
    record = consent()
    record["models"]["rf_detr_2xl"][field] = value
    save(runtime, record)
    assert not model_licenses.is_model_license_accepted("rf_detr_2xl")


@pytest.mark.parametrize("payload", [None, [], {"schema_version": True, "models": {}}, {"schema_version": 1, "models": []}])
def test_malformed_registry_is_not_consent(runtime, payload):
    save(runtime, payload)
    assert not model_licenses.is_model_license_accepted("rf_detr_2xl")


def test_unreadable_or_oversized_records_fail_closed(runtime):
    path = runtime / "model_license_acceptance.json"
    path.write_bytes(b"invalid JSON")
    assert not model_licenses.is_model_license_accepted("rf_detr_2xl")
    path.write_bytes(b" " * (64 * 1024 + 1))
    assert not model_licenses.is_model_license_accepted("rf_detr_2xl")


def test_default_record_location_is_app_runtime_when_no_override(runtime, monkeypatch):
    monkeypatch.delenv("VISION_WORKBENCH_RUNTIME")
    monkeypatch.setattr(model_licenses, "APP_ROOT", runtime)
    assert model_licenses.license_record_path() == runtime / "runtime" / "model_license_acceptance.json"
