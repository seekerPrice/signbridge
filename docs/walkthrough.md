# SignBridge — technical walkthrough

> Internal technical record of the build. Not a submission deliverable
> (Build-in-Public extra challenge was dropped on 2026-05-07).
> Kept around because it documents the AMD-specific engineering thinking
> and is useful if anyone later asks "why these design choices?".

## What we built

A real-time webcam-based ASL → English speech translator. A deaf user signs
into the webcam; the pipeline (MediaPipe Holistic → trained sign classifier
→ Llama-3.1-8B sentence composer → Coqui XTTS-v2) returns spoken English
in under 2 seconds. Designed to fit Track 3 (Vision & Multimodal AI) with
the entire model stack running concurrently on a single AMD Instinct MI300X.

## Why AMD MI300X

- 192 GB HBM3 — the trained classifier (~20 MB), Llama-3.1-8B (~16 GB FP16),
  XTTS-v2 (~2 GB), and (V2 stretch) Whisper-large-v3 (~3 GB) all fit
  concurrently with margin for KV cache.
- 5.3 TB/s memory bandwidth — bandwidth-bound streaming workload (many
  small inferences per second on the classifier + TTS chunked decode + LLM
  next-token) is exactly what bandwidth wins.

## Architecture

```
webcam frames → MediaPipe Holistic → trained classifier
                  (CPU-fast)            (TorchScript on MI300X)
                                              │
                                              ▼
                                  Llama-3.1-8B sentence composer
                                       (vLLM on MI300X)
                                              │
                                              ▼
                                          XTTS-v2 → audio
                                       (XTTS on MI300X)
```

## Models

| Component | Source | Notes |
|---|---|---|
| Pose extractor | MediaPipe Holistic (Google) | CPU-fast preprocessing — not GPU-bound |
| Sign classifier | trained from scratch on WLASL Top-100 + ASL fingerspelling alphabet | 3-layer transformer encoder over 543-dim landmark sequences; published to HF Hub at `lucas-loo/signbridge-classifier` |
| Sentence composer | `meta-llama/Llama-3.1-8B-Instruct` | Pulled from HF Hub; served on MI300X via vLLM |
| Text-to-speech | `coqui/XTTS-v2` | Multilingual; we use English V1 |

## Datasets

- [WLASL](https://github.com/dxli94/WLASL) Top-100 subset
- ASL fingerspelling alphabet (open dataset)

## ROCm / AMD Developer Cloud experience

> *Filled in across Day 1–3.*

### Day 1 — environment + sanity
TODO

### Day 2 — training the classifier
TODO

### Day 3 — serving + latency tuning
TODO

### What worked well
TODO

### What we'd flag as friction
TODO

## Latency

Target: ≤ 2 s from end-of-sign to start of speech.

Measured on a single MI300X (Day 3):
- MediaPipe Holistic per frame: TODO ms
- Classifier per window: TODO ms
- Llama-3.1-8B sentence composition (≤ 30 tokens): TODO ms
- XTTS-v2 first-audio-chunk: TODO ms

## MI300X vs NVIDIA H100 — the AMD pitch

| Item | MI300X (1 GPU) | H100 (1 GPU) | H100 cluster needed |
|---|---|---|---|
| Llama-3.1-8B FP16 weights | ✅ fits with margin | ✅ fits with margin | 1× |
| + XTTS-v2 + Whisper-large-v3 + classifier | ✅ all concurrent | ⚠️ tight (~28 GB total + KV) | likely 1× but no headroom |
| + 70B reasoner upgrade (V2) | ✅ 70B FP8 ~70 GB still fits | ❌ doesn't fit at all | ≥3× |

The single-GPU concurrency story is the AMD pitch. This V1 fits on H100;
the architecture has clear headroom on MI300X for higher-quality V2 models.

## License

Apache 2.0.
