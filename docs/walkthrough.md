# SignBridge — technical walkthrough

> Internal technical record of the build. Not a submission deliverable
> (Build-in-Public extra challenge was dropped on 2026-05-07).
> Kept around because it documents the AMD-specific engineering thinking
> and is useful if anyone later asks "why these design choices?".

## What we built

A real-time webcam-based ASL → English speech translator. A deaf user signs
into the webcam; the pipeline (MediaPipe Hand → trained MLP for static
fingerspelling, OR webcam-clip → ffmpeg → fine-tuned Qwen3-VL-8B native
video → Qwen3-8B composer → gTTS) returns spoken English in under 2
seconds. Designed to fit Track 3 (Vision & Multimodal AI) with both LLMs
running concurrently on a single AMD Instinct MI300X.

## Why AMD MI300X

- 192 GB HBM3 — the trained MLP classifier (~478 KB), fine-tuned
  Qwen3-VL-8B (~16 GB FP16), Qwen3-8B composer (~16 GB FP16), and
  (V2 stretch) Whisper-large-v3 (~3 GB) all fit concurrently with margin
  for KV cache.
- 5.3 TB/s memory bandwidth — bandwidth-bound streaming workload (many
  small inferences per second on the MLP + LLM next-token + Qwen3-VL
  vision encoder) is exactly what bandwidth wins.

## Architecture

```
Snapshot tab (fingerspelling):
  webcam frame → MediaPipe Hand → trained MLP classifier
                   (CPU-fast)        (PyTorch on CPU, ~50 ms)

Record sign tab (motion words):
  webcam recording → ffmpeg (480p, 8 fps, ≤4 s, H.264)
                          ↓
                   vLLM video_url block on AMD MI300X port 8000
                          ↓
              fine-tuned Qwen3-VL-8B (native video understanding)

Both paths converge:
                                   ↓
                          Qwen3-8B sentence composer
                          (vLLM on MI300X port 8001)
                                   ↓
                                  gTTS
                          (Google free TTS, MP3)
```

## Models

| Component | Source | Notes |
|---|---|---|
| Hand-pose extractor | MediaPipe HandLandmarker (Google) | CPU-only, ~50ms/frame — runs on the HF Space CPU |
| Static-letter classifier (Snapshot tab) | **trained-from-scratch MLP** on hand-landmark vectors → 26 ASL letters | 3-layer MLP (63→256→256→128→26), 5K trainable params, GELU+dropout. **88.0% test accuracy** on a 1,727-image holdout, **90.4% on the gold set**. Weights at `huggingface.co/LucasLooTan/signbridge-asl-classifier` |
| Motion-sign + fallback recognizer | **fine-tuned `Qwen/Qwen3-VL-8B-Instruct`** | LoRA fine-tune on AMD MI300X (rank 16, target q/k/v/o, 2 epochs, 54 min wall-clock on a single MI300X). Eval loss 0.48, transformers gold-set accuracy 92.3%. Merged adapter pushed to `huggingface.co/LucasLooTan/signbridge-qwen3vl-8b-asl` (17.5 GB) |
| Sentence composer | `Qwen/Qwen3-8B` | Pulled from HF Hub; served on MI300X via vLLM. Used for every Speak click — AMD is in the critical path |
| Text-to-speech | `gTTS` (Google's free TTS) | Tiny dependency, no model download, MP3 output in <1 s. Coqui XTTS-v2 path is preserved as Tier 1 fallback when installed locally |

## Datasets

- **Marxulia/asl_sign_languages_alphabets_v03** (HF Hub) — 10,873 photographic ASL letter samples; we extracted MediaPipe landmarks (8,639 hands detected) + used the same images for the LoRA fine-tune (9,786 train / 1,087 eval split)
- [WLASL](https://github.com/dxli94/WLASL) Top-100 subset — referenced for V2 motion-sign training (not used in V1)

## ROCm / AMD Developer Cloud experience

### Day 1 — environment + sanity
- Provisioned an MI300X-1× droplet (192 GB HBM3, 240 GB RAM, 5 TB scratch) at $1.99/hr in ATL1
- Selected the prebuilt **vLLM 0.17.1 / ROCm 7.2** Quick-Start image — saved ~30 min vs hand-installing
- ROCm reported the GPU correctly via `rocm-smi`; vLLM spun up Qwen3-VL-32B + Qwen3-8B in parallel within 12 minutes
- One real friction: vLLM's default `0.0.0.0` binding tripped a Gloo/NCCL error on the host's NIC; fixed by setting `VLLM_HOST_IP=127.0.0.1` and `GLOO_SOCKET_IFNAME=lo` env vars

### Day 2 — fine-tuning Qwen3-VL-8B with LoRA on MI300X
- Used the AMD-provided `rocm:latest` Docker image — torch 2.9.1+ROCm, transformers 4.57.6, peft 0.18.1, accelerate 1.13.0 all preinstalled
- LoRA rank 16 on q/k/v/o projections, FP16, gradient checkpointing with `use_reentrant=False`
- Critical fix for PEFT + grad-checkpoint: call `model.enable_input_require_grads()` BEFORE wrapping in PEFT (without it, training stalls at step 0 with "None of the inputs have requires_grad=True")
- 1,224 steps × 4×4 effective batch = 9,786 samples × 2 epochs in 54 minutes; eval loss 0.48
- Spent ~$2 of the $100 credit on this single fine-tune

### Day 3 — serving + accuracy comparison
- **Three approaches benchmarked on the same 52-image gold set:**
  - Qwen3-VL-32B zero-shot: **19.2%** — VLMs without ASL-specific tuning struggle with subtle hand shapes
  - MediaPipe + 5K-param MLP: **90.4%** — the textbook approach for static pose classification still wins for cost/accuracy ratio
  - LoRA-tuned Qwen3-VL-8B (transformers eval): **92.3%** — best, but 4× slower per inference
- Hybrid pipeline ships: MediaPipe+MLP for typical fingerspelling (50ms, ~90%), fine-tuned VLM for motion signs and as fallback when no hand is detected
- Latency on MI300X: Qwen3-8B composer ~0.5s/call, fine-tuned 8B vision recognizer ~1.3s/call

### What worked well
- AMD Developer Cloud provisioning was 5 min from "approved" to SSH — credit landed via email and the Quick-Start vLLM image meant zero ROCm setup pain
- 192 GB HBM3 hosted both the 32B vision model and the 8B composer concurrently (gpu-mem 0.55 + 0.30) with margin for KV cache
- Fine-tuning + inference + composing on a single MI300X with no swapping or reloading — the multi-tenant story is real
- The `rocm:latest` Docker image had the entire training stack (torch, transformers, peft, accelerate, datasets) preinstalled and tested

### What we'd flag as friction
- vLLM 0.17.1's image-preprocessing for Qwen3-VL doesn't exactly match transformers' processor — the LoRA-tuned model that scored 92.3% in transformers eval drops to 63.5% via the OpenAI-compatible vLLM endpoint. This is upstream and not AMD-specific, but it limited how aggressively we could lean on the fine-tune for the live demo
- The `low-power state` warning in `rocm-smi` while the GPU was idle was cosmetic but confusing — clarifying that "low-power" doesn't mean "stalled" would help first-time users
- Setting `VLLM_HOST_IP=127.0.0.1` for single-GPU vLLM on a Gloo backend isn't documented in the AMD vLLM Quick-Start; we found it from a vLLM GitHub issue

### What we'd flag as friction
TODO

## Why AMD MI300X — concretely

The pipeline (MediaPipe Hand + fine-tuned Qwen3-VL-8B + Qwen3-8B composer + gTTS)
fits comfortably on a single MI300X with KV-cache headroom. The same workload
on NVIDIA forces sharding once we add the V2 reasoner.

| Component | Weights (FP16) | MI300X 1× (192 GB) | H100 80 GB | H200 141 GB |
|---|---|---|---|---|
| Fine-tuned Qwen3-VL-8B (vision, native video) | ~16 GB | ✅ fits | ✅ | ✅ |
| Qwen3-8B (composer) | ~16 GB | ✅ fits | ✅ | ✅ |
| Whisper-large-v3 (V2 reverse direction) | ~3 GB | ✅ fits | ⚠ tight | ✅ |
| gTTS (no GPU footprint — Python-side cloud call) | n/a | ✅ | ✅ | ✅ |
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
- MediaPipe Hand detection per frame: ~50 ms (CPU)
- Trained MLP per landmark vector: ~5 ms (CPU)
- Fine-tuned Qwen3-VL-8B per recording (native video, ~1680 prompt tokens): ~1-2 s
- Qwen3-8B sentence composition (≤ 30 tokens): ~300 ms
- gTTS first-audio-chunk: ~500 ms (single round-trip to Google)

## MI300X vs NVIDIA H100 — the AMD pitch

| Item | MI300X (1 GPU) | H100 (1 GPU) | H100 cluster needed |
|---|---|---|---|
| Fine-tuned Qwen3-VL-8B + Qwen3-8B (both FP16) | ✅ fits with margin | ⚠️ tight (~32 GB) | maybe 1×, no headroom |
| + Whisper-large-v3 + MLP classifier | ✅ all concurrent | ⚠️ tight (~35 GB total + KV) | likely 1× but no headroom |
| + 70B reasoner upgrade (V2) | ✅ 70B FP8 ~70 GB still fits | ❌ doesn't fit at all | ≥3× |

The single-GPU concurrency story is the AMD pitch. This V1 fits on H100;
the architecture has clear headroom on MI300X for higher-quality V2 models.

## License

MIT.
