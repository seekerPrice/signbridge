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

## Why AMD MI300X — concretely

The pipeline (MediaPipe Holistic + Qwen3-VL-8B + Llama-3.1-8B + Coqui XTTS-v2)
fits comfortably on a single MI300X with KV-cache headroom. The same workload
on NVIDIA forces sharding once we add the V2 reasoner.

| Component | Weights (FP16) | MI300X 1× (192 GB) | H100 80 GB | H200 141 GB |
|---|---|---|---|---|
| Qwen3-VL-8B (vision) | ~16 GB | ✅ fits | ✅ | ✅ |
| Llama-3.1-8B (composer) | ~16 GB | ✅ fits | ✅ | ✅ |
| Whisper-large-v3 (V2 reverse direction) | ~3 GB | ✅ fits | ⚠ tight | ✅ |
| Coqui XTTS-v2 (TTS) | ~2 GB | ✅ fits | ⚠ tight | ✅ |
| (V2) Llama-3.1-70B FP8 reasoner upgrade | ~70 GB | ✅ still fits | ❌ doesn't fit at all | ⚠ FP8 only, no headroom |
| **Concurrent serving + KV cache** | ✅ comfortable | ❌ requires sharding | ⚠ tight | ✅ |

The single-GPU concurrency story is the AMD pitch. V1 fits anywhere; the
architecture has clear MI300X headroom for V2 model upgrades that NVIDIA
H100 cannot match without sharding across multiple cards.

## Deployment ethics

SignBridge is a *substrate*, not a finished product. We ship the open-source
multi-modal pipeline so Deaf-led organisations — schools-for-the-Deaf, regional
NGOs, ministries of social services — can deploy on their own AMD compute,
fine-tune for their dialect, and own the deployment.

Three principles, drawn from the Deaf-led literature on sign-language AI:

1. **ASL only V1** is a scope decision. Sign languages are not interchangeable
   — BSL, ISL, MSL, CSL, and 200+ other sign languages each deserve their own
   teams, training data, and Deaf community leadership. Bragg et al.
   ["Systemic Biases in Sign Language AI Research"](https://arxiv.org/html/2403.02563v1)
   (2024, Deaf-led position paper) is direct on this point.

2. **Deaf community engagement before deployment.** Per the ACM ASSETS 2025
   ["Exploring Collaboration to Center the Deaf Community in Sign Language AI"](https://dl.acm.org/doi/10.1145/3663547.3746390)
   the productive ML/Deaf collaboration question isn't "how do we build this?"
   but "*should* we build this, *for whom*, *with whom*?". Any deployment
   downstream of this code must answer that locally.

3. **Privacy by default.** SignBridge sessions are ephemeral — webcam frames
   and audio are processed in-memory and not persisted server-side beyond the
   request lifetime. In the spirit of [Privacy-Aware Sign Language Translation
   at Scale](https://aclanthology.org/2024.acl-long.467.pdf) (ACL 2024).

## Future work — academic foundations we'd build on next

- **SignCLIP** ([EMNLP 2024](https://aclanthology.org/2024.emnlp-main.518.pdf)) —
  learned text↔sign embeddings; replaces the prompt-only composer with a
  CLIP-style alignment head for higher-quality sign-to-English mapping.
- **SL-SLR** ([arXiv 2509.05188v1](https://arxiv.org/html/2509.05188v1)) —
  self-supervised representation learning with motion-aware data augmentation;
  the right path if we ever train a custom classifier on raw signer footage.
- **Continuous SLT trained models** (Swin-MSTP, Stack Transformer) — the
  current trained-from-scratch ceiling on WLASL is ~93.5% Top-1. The VLM
  zero-shot path we ship here is a *deployment-cost* play, not an
  accuracy-ceiling play; SignCLIP-style learned embeddings are the natural
  V2 step toward that ceiling.

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

MIT.
