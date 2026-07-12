"""Tests for cjm_capability_voxtral_hf.capability — pure-compute Voxtral tool.

Projected from the capability notebook's test cells at the c25780e8 flip
(hermetic: initialize is lazy config-apply, no model download; the transcribe
path is exercised by the e2e harnesses)."""
from dataclasses import fields

import pytest

from cjm_capability_voxtral_hf.capability import (VoxtralHFCapability,
                                                  VoxtralHFCapabilityConfig)
from cjm_substrate.core.capability import ToolCapability
from cjm_substrate.utils.validation import SCHEMA_ENUM, dict_to_config


def test_pure_compute_surface():
    capability = VoxtralHFCapability()
    assert isinstance(capability, ToolCapability)
    assert capability.config_class.__name__ == "VoxtralHFCapabilityConfig"
    assert capability.version
    # native-surface model: pure-compute transcribe replaces the fused execute
    assert hasattr(capability, "transcribe") and not hasattr(capability, "execute")
    assert not hasattr(capability, "supported_formats")
    # streaming override retired (descoped GUI host) — the OVERRIDE is gone from
    # the class __dict__; the base ToolCapability still provides the default
    assert "execute_stream" not in VoxtralHFCapability.__dict__
    assert "supports_streaming" not in VoxtralHFCapability.__dict__


def test_model_enum_and_validation():
    model_field = next(f for f in fields(VoxtralHFCapabilityConfig) if f.name == "model_id")
    models = model_field.metadata.get(SCHEMA_ENUM, [])
    assert "mistralai/Voxtral-Mini-3B-2507" in models

    cfg = dict_to_config(VoxtralHFCapabilityConfig,
                         {"model_id": "mistralai/Voxtral-Mini-3B-2507"}, validate=True)
    assert cfg.model_id == "mistralai/Voxtral-Mini-3B-2507"
    with pytest.raises(ValueError):
        dict_to_config(VoxtralHFCapabilityConfig, {"model_id": "invalid_model"}, validate=True)
    with pytest.raises(ValueError):
        dict_to_config(VoxtralHFCapabilityConfig,
                       {"model_id": "mistralai/Voxtral-Mini-3B-2507", "temperature": 2.5},
                       validate=True)


def test_initialize_is_lazy_config_apply():
    capability = VoxtralHFCapability()
    capability.initialize({"model_id": "mistralai/Voxtral-Mini-3B-2507", "device": "cpu"})
    current = capability.get_current_config()
    assert isinstance(current, dict)
    assert current["model_id"] == "mistralai/Voxtral-Mini-3B-2507"


def test_config_schema_for_ui():
    schema = VoxtralHFCapability().get_config_schema()
    assert schema["name"] == "VoxtralHFCapabilityConfig"
    assert len(schema["properties"]) == len(fields(VoxtralHFCapabilityConfig))
    assert schema["properties"]["model_id"].get("enum")
