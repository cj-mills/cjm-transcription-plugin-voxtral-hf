"""Capability implementation for Mistral Voxtral transcription through Hugging Face Transformers (Option C, stage 8: pure compute).

Stage 8 (Option C / PILLAR 1c): the tool re-bases onto ToolCapability (pure
compute). The cache/persist bookends + the TranscriptionResult data noun moved
OUT — the generic adapter (cjm-transcription-adapter-interface) owns the cache,
and the result DTO lives in cjm-capability-primitives. No get_plugin_metadata.
Shared helpers: cjm-substrate-torch-utils (release + CUDA-OOM typing) and
cjm-substrate-hf-utils (cache-config mixin + progress download + OOM-typed load).
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

import torch
from cjm_capability_primitives.transcription import TranscriptionResult
from cjm_substrate.core.capability import EnvVarSpec, RELOAD_TRIGGER, ToolCapability
from cjm_substrate.core.errors import (CapabilityFatalError, CapabilityInputError,
                                       CapabilityResourceError)
from cjm_substrate.utils.validation import (config_to_dict, dataclass_to_jsonschema, dict_to_config,
                                            SCHEMA_DESC, SCHEMA_ENUM, SCHEMA_MAX, SCHEMA_MIN,
                                            SCHEMA_TITLE)
from cjm_substrate_hf_utils.cache_config import HFCacheConfig
from cjm_substrate_hf_utils.download import snapshot_download_with_progress
from cjm_substrate_hf_utils.loading import load_pretrained_with_oom
from cjm_substrate_torch_utils.memory import release_model
from cjm_substrate_torch_utils.oom import cuda_oom_to_capability_resource_error

try:
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    VOXTRAL_AVAILABLE = True
except ImportError:
    VOXTRAL_AVAILABLE = False


@dataclass
class VoxtralHFCapabilityConfig(HFCacheConfig):
    """Configuration for Voxtral HF transcription capability."""
    model_id:str = field(
        default="mistralai/Voxtral-Mini-3B-2507",
        metadata={
            SCHEMA_TITLE: "Model ID",
            RELOAD_TRIGGER: "model",  # CR-4: change triggers model reload
            SCHEMA_DESC: "Voxtral model to use. Mini is faster, Small is more accurate.",
            SCHEMA_ENUM: ["mistralai/Voxtral-Mini-3B-2507", "mistralai/Voxtral-Small-24B-2507"]
        }
    )
    device:str = field(
        default="auto",
        metadata={
            SCHEMA_TITLE: "Device",
            RELOAD_TRIGGER: "model",  # CR-4: change triggers model reload
            SCHEMA_DESC: "Device for inference (auto will use CUDA if available)",
            SCHEMA_ENUM: ["auto", "cpu", "cuda"]
        }
    )
    dtype:str = field(
        default="auto",
        metadata={
            SCHEMA_TITLE: "Data Type",
            RELOAD_TRIGGER: "model",  # CR-4: change triggers model reload
            SCHEMA_DESC: "Data type for model weights (auto uses bfloat16; set float32 explicitly for full precision)",
            SCHEMA_ENUM: ["auto", "bfloat16", "float16", "float32"]
        }
    )
    language:Optional[str] = field(
        default="en",
        metadata={
            SCHEMA_TITLE: "Language",
            SCHEMA_DESC: "Language code for transcription (e.g., 'en', 'es', 'fr')"
        }
    )
    max_new_tokens:int = field(
        default=25000,
        metadata={
            SCHEMA_TITLE: "Max New Tokens",
            SCHEMA_DESC: "Maximum number of tokens to generate",
            SCHEMA_MIN: 1,
            SCHEMA_MAX: 50000
        }
    )
    do_sample:bool = field(
        default=False,
        metadata={
            SCHEMA_TITLE: "Do Sample",
            SCHEMA_DESC: "Whether to use sampling (true) or greedy decoding (False)"
        }
    )
    temperature:float = field(
        default=1.0,
        metadata={
            SCHEMA_TITLE: "Temperature",
            SCHEMA_DESC: "Temperature for sampling (only used when do_sample=true)",
            SCHEMA_MIN: 0.0,
            SCHEMA_MAX: 2.0
        }
    )
    top_p:float = field(
        default=0.95,
        metadata={
            SCHEMA_TITLE: "Top P",
            SCHEMA_DESC: "Top-p (nucleus) sampling parameter (only used when do_sample=true)",
            SCHEMA_MIN: 0.0,
            SCHEMA_MAX: 1.0
        }
    )
    compile_model:bool = field(
        default=False,
        metadata={
            SCHEMA_TITLE: "Compile Model",
            SCHEMA_DESC: "Use torch.compile for potential speedup (requires PyTorch 2.0+)"
        }
    )
    load_in_8bit:bool = field(
        default=False,
        metadata={
            SCHEMA_TITLE: "Load in 8-bit",
            RELOAD_TRIGGER: "model",  # CR-4: change triggers model reload
            SCHEMA_DESC: "Load model in 8-bit quantization (requires bitsandbytes)"
        }
    )
    load_in_4bit:bool = field(
        default=False,
        metadata={
            SCHEMA_TITLE: "Load in 4-bit",
            RELOAD_TRIGGER: "model",  # CR-4: change triggers model reload
            SCHEMA_DESC: "Load model in 4-bit quantization (requires bitsandbytes)"
        }
    )


class VoxtralHFCapability(ToolCapability):
    """Mistral Voxtral transcription capability via Hugging Face Transformers (stage 8: pure-compute tool capability).

    Native-surface model (PILLAR 1c): this tool is PURE COMPUTE — `transcribe`
    loads the model, runs inference, and builds the typed `TranscriptionResult`.
    The cache-check + persistence bookends + the per-call `force` control live in
    the generic transcription adapter (cjm-transcription-adapter-interface); the
    result DTO lives in cjm-capability-primitives; identity is derived from the
    installed distribution. No `get_plugin_metadata`, no `self.storage`."""

    # CR-4: declarative reload-triggers — substrate's reconfigure_with_triggers
    # walks this config_class's dataclass fields for RELOAD_TRIGGER metadata and
    # fires the corresponding `_release_<trigger>` method on field changes.
    config_class = VoxtralHFCapabilityConfig

    # Track 19 (CR-12 worker-env model): worker spawn env declared on the class.
    # CUDA_VISIBLE_DEVICES + OMP_NUM_THREADS are static; HF_HOME is templated to
    # the substrate models dir. The substrate resolves + injects at Popen.
    WORKER_ENV: ClassVar[List[EnvVarSpec]] = [
        EnvVarSpec(
            name="CUDA_VISIBLE_DEVICES",
            default="0",
            label="GPU Device",
            description="Which GPU index the worker uses.",
        ),
        EnvVarSpec(
            name="OMP_NUM_THREADS",
            default="4",
            label="OpenMP Threads",
            description="Thread cap for CPU-side ops.",
        ),
        EnvVarSpec(
            name="HF_HOME",
            default="${CJM_MODELS_DIR}/huggingface",
            label="HF Cache Directory",
            description="HuggingFace Hub cache root (templated to the substrate models dir).",
        ),
    ]

    def __init__(self):
        """Initialize the Voxtral HF capability with default configuration."""
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.config: VoxtralHFCapabilityConfig = None
        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None

    @property
    def name(self) -> str: # Capability name identifier
        """Capability identity, derived from the installed distribution (PILLAR 1c).

        Runtime-derived: in the worker / in-env introspection `__package__`
        resolves; the manifest records the same value independently (the
        dual-mode generator reads it from the distribution)."""
        from importlib.metadata import metadata, packages_distributions
        dist = (packages_distributions().get(__package__) or [__package__.replace("_", "-")])[0]
        return metadata(dist)["Name"]

    @property
    def version(self) -> str: # Capability version string
        """Get the capability version string."""
        from cjm_capability_voxtral_hf import __version__
        return __version__

    def get_current_config(self) -> Dict[str, Any]: # Current configuration as dictionary
        """Return current configuration state."""
        if not self.config:
            return {}
        return config_to_dict(self.config)

    def get_config_schema(self) -> Dict[str, Any]: # JSON Schema for configuration
        """Return JSON Schema for UI generation."""
        return dataclass_to_jsonschema(VoxtralHFCapabilityConfig)

    @staticmethod
    def get_config_dataclass() -> VoxtralHFCapabilityConfig: # Configuration dataclass
        """Return dataclass describing the capability's configuration options."""
        return VoxtralHFCapabilityConfig

    def initialize(
        self,
        config: Optional[Any] = None # Configuration dataclass, dict, or None
    ) -> None:
        """First-time setup. CR-4: the manual model/device/dtype/quantization
        diff-and-reload is replaced by declarative RELOAD_TRIGGER metadata; the
        substrate's reconfigure path fires _release_model then re-applies config."""
        self._apply_config(config)
        self.logger.info(f"Initialized Voxtral HF capability with model '{self.config.model_id}' on device '{self.device}' with dtype '{self.dtype}'")

    def transcribe(
        self,
        audio: Union[str, Path], # Path to MODEL-READY audio (converted upstream)
        **kwargs # Provenance (source_start_time/source_end_time) stamped into metadata
    ) -> TranscriptionResult: # Typed transcription output
        """Transcribe model-ready audio using Voxtral — PURE COMPUTE.

        Stage 8 / PILLAR 1c: the cache-check + persistence bookends moved to the
        generic transcription adapter; this method loads the model, runs
        inference, and builds the typed result. Model params come from
        `self.config` (the CR-15 per-call override path is gone — the tool runs
        its effective config, no metadata lie); `source_start_time` /
        `source_end_time` ride the provenance kwarg channel into metadata."""
        # Validate + resolve the input path, then load the model.
        audio_path = self._prepare_audio(audio)
        self._load_model()

        # Effective config (no per-call override path).
        c = self.config
        model_id = c.model_id
        language = c.language
        max_new_tokens = c.max_new_tokens
        do_sample = c.do_sample
        temperature = c.temperature
        top_p = c.top_p

        # Prepare inputs
        self.logger.info(f"Processing audio with Voxtral {model_id}")

        inputs = self.processor.apply_transcription_request(
            language=language or "en",
            audio=str(audio_path),
            model_id=model_id
        )
        inputs = inputs.to(self.device, dtype=self.dtype)

        # Generation kwargs
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        # Add sampling parameters if sampling is enabled
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        # Generate transcription. SG-47 Track B wraps the inference site so
        # CUDA OOM surfaces as CapabilityResourceError → CR-7 reactive-retry reloads.
        try:
            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    outputs = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )
        except torch.cuda.OutOfMemoryError as e:
            raise cuda_oom_to_capability_resource_error(
                e, label=f"Voxtral inference (model={model_id!r})",
            ) from e

        # Decode the output
        result_text = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        # Clean up tensors immediately
        del inputs
        del outputs
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Capture provenance metadata passed via kwargs
        provenance_meta = {
            k: v for k, v in kwargs.items()
            if k in ['source_start_time', 'source_end_time']
        }

        # Create transcription result
        transcription_result = TranscriptionResult(
            text=result_text.strip(),
            confidence=None,  # Voxtral doesn't provide confidence scores
            segments=None,  # Voxtral doesn't provide segments by default
            metadata={
                "model": model_id,
                **provenance_meta,
                "language": language or "en",
                "device": self.device,
                "dtype": str(self.dtype),
            }
        )

        self.logger.info(f"Transcription completed: {len(result_text.split())} words")
        return transcription_result

    def _apply_config(
        self,
        config: Optional[Any] = None # Configuration dataclass, dict, or None
    ) -> None:
        """CR-4: apply config + derive config-dependent state (device, dtype). No
        heavy-resource work. Called by initialize (first-time) and the substrate's
        reconfigure delta path. Model release on a model_id/device/dtype/quantization
        change is handled declaratively via RELOAD_TRIGGER -> _release_model."""
        self.config = dict_to_config(VoxtralHFCapabilityConfig, config or {})

        # Resolve device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        # Resolve dtype. G9: `auto` resolves to bfloat16 UNIFORMLY (device-independent).
        # The prior auto->float32-on-CPU default doubled the CPU footprint — stage-3 G9:
        # Voxtral-Small-24B with auto on CPU hit ~83GB RSS + swap and was unusable, while
        # explicit bfloat16 on CPU ran fine. Full precision stays available via an
        # explicit dtype="float32".
        if self.config.dtype == "auto":
            self.dtype = torch.bfloat16
        else:
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32
            }
            self.dtype = dtype_map[self.config.dtype]

    def _release_model(self) -> None:
        """Unload the current model + processor and free GPU memory.

        Delegates to cjm-substrate-torch-utils' `release_model` (move-to-CPU / del / gc /
        empty_cache / synchronize) -- the single source of truth across torch GPU capabilitys."""
        if self.model is None and self.processor is None:
            return
        self.logger.info("Unloading Voxtral model for reconfiguration")
        release_model(self, ["model", "processor"], self.device or "cuda", logger=self.logger)

    def _load_model(self) -> None:
        """Load the Voxtral model + processor (lazy).

        The heartbeat wraps BOTH the (potentially long, often quiet) snapshot download
        AND the silent from_pretrained build, so the substrate's prefetch stall detector
        always sees the (progress, message) tuple advance. snapshot_download_with_progress
        layers real per-file download % on top when the HF Hub tqdm callback fires.
        CUDA OOM on load surfaces as a typed CapabilityResourceError for CR-7 reactive retry."""
        if self.model is not None and self.processor is not None:
            return
        self.logger.info(f"Loading Voxtral model: {self.config.model_id}")

        # Built before the heartbeat block (instant). The snapshot below guarantees the
        # cache is populated, so the loads run local-only.
        model_kwargs = {
            "cache_dir": self.config.cache_dir,
            "revision": self.config.revision,
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": True,
            "device_map": self.device,
        }
        if self.config.load_in_8bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            model_kwargs["dtype"] = self.dtype

        try:
            with self.heartbeat(f"loading Voxtral model {self.config.model_id}"):
                # Download (honors air-gap via local_files_only). The heartbeat is the
                # floor here; the tqdm hook adds real "downloading <file>" % when it fires.
                # G6: skip the Mistral original-format `consolidated.safetensors` — on
                # Voxtral-Small-24B it is a 48.5GB second full copy of the weights (Mini
                # ships an 8.75GB one) that HF `from_pretrained` never loads (it reads the
                # sharded `model-*.safetensors` via `model.safetensors.index.json`). The
                # `ignore_patterns` rides snapshot_download's **kwargs passthrough (no
                # shared-util change) and no-ops on repos without a consolidated copy.
                snapshot_download_with_progress(
                    self.config.model_id,
                    report_progress=self.report_progress,
                    cache_dir=self.config.cache_dir,
                    revision=self.config.revision,
                    local_files_only=self.config.local_files_only,
                    ignore_patterns=["consolidated*"],
                )
                self.processor = AutoProcessor.from_pretrained(
                    self.config.model_id,
                    cache_dir=self.config.cache_dir,
                    revision=self.config.revision,
                    trust_remote_code=self.config.trust_remote_code,
                    local_files_only=True,
                )
                self.model = load_pretrained_with_oom(
                    VoxtralForConditionalGeneration,
                    self.config.model_id,
                    label=f"loading Voxtral model {self.config.model_id!r}",
                    **model_kwargs,
                )

            if self.config.compile_model and hasattr(torch, "compile"):
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled with torch.compile")
            self.logger.info("Voxtral model loaded successfully")
        except CapabilityResourceError:
            raise  # already typed by load_pretrained_with_oom
        except torch.cuda.OutOfMemoryError as e:
            # Defensive: OOM outside the wrapped model load (processor / compile).
            raise cuda_oom_to_capability_resource_error(
                e, label=f"loading Voxtral model {self.config.model_id!r}",
            ) from e
        except Exception as e:
            raise CapabilityFatalError(f"Failed to load Voxtral model: {e}") from e

    def _prepare_audio(
        self,
        audio: Union[str, Path] # Path to a decodable audio file
    ) -> str: # The audio file path
        """Validate the audio input and return it as a path string.

        The caller (orchestration / proxy) guarantees a model-ready audio file;
        in-memory preparation is no longer a capability responsibility."""
        if isinstance(audio, (str, Path)):
            return str(audio)
        raise CapabilityInputError(  # SG-47: typed input-validation (multi-inherits ValueError)
            f"Unsupported audio input type: {type(audio)}; expected a file path (str or Path)",
            fields_invalid=["audio"],
        )

    def is_available(self) -> bool: # True if Voxtral and its dependencies are available
        """Check if Voxtral is available."""
        return VOXTRAL_AVAILABLE

    def prefetch(self) -> None:
        """CR-4 (SG-19): eagerly load the model + processor so the first execute()
        doesn't pay the download/load cost. Idempotent via _load_model's None-guard."""
        self._load_model()

    def on_disable(self) -> None:
        """CR-2: release the GPU model + processor when the operator disables the
        capability (the worker stays alive); lazy reload on the next execute."""
        self._release_model()

    def cleanup(self) -> None:
        """Release the model + processor (CR-4: delegates to `_release_model`)."""
        self._release_model()
