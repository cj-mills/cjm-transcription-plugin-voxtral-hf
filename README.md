# cjm-capability-voxtral-hf

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

A Mistral Voxtral speech-to-text capability for the cjm-substrate runtime that provides local transcription through Hugging Face Transformers with configurable model selection and parameter control.

## Modules

- **`cjm_capability_voxtral_hf.capability`**

## API

### `cjm_capability_voxtral_hf.capability`

- `VoxtralHFCapability` _class_ — Mistral Voxtral transcription capability via Hugging Face Transformers (stage 8: pure-compute tool capability).
- `VoxtralHFCapabilityConfig` _class_ — Configuration for Voxtral HF transcription capability.

## Dependencies

**Depends on:** `cjm-capability-primitives`, `cjm-substrate`, `hf-utils`, `torch-utils`
