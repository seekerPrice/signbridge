# SignBridge — real-time ASL → speech translation

Loaded when the working directory is inside `/Users/lucaslt/Documents/side-gig/amd-hackathon/`. Keep this file current: prepend a dated entry to the Progress log after every milestone. Prune entries older than 60 days unless they anchor a persistent fact.

---

## Standing rules

- **Never make assumptions — always look up answers online.** Before coding, configuring, or recommending anything, verify against authoritative sources (use `context7` for libraries / SDKs / APIs, `WebSearch` / `WebFetch` for everything else). Training data is stale; default-guesses waste time. This applies even to things that "seem obvious".
- **Use Superpowers skills for every suitable use case — especially planning.** Any planning, debugging, executing-from-plan, brainstorming, parallel-agent dispatch, TDD, or pre-completion verification goes through the matching `superpowers:*` skill (`superpowers:writing-plans`, `:executing-plans`, `:brainstorming`, `:systematic-debugging`, `:subagent-driven-development`, `:verification-before-completion`, `:test-driven-development`, `:dispatching-parallel-agents`). Free-form prose plans are not allowed.
- **Use the `deep-research` skill for deep academic research.** Multi-source comparison, literature review, state-of-the-art surveys, citation-tracked evidence — invoke `deep-research`, not ad-hoc web search.

---

## Status

Day 1 / ~4 — pivoted from Iris to SignBridge on 2026-05-07. **Submission deadline: 2026-05-11 03:00 MYT.** ~3.5 days remaining. AMD Developer Hackathon, **Track 3 — Vision & Multimodal AI** (only — Build-in-Public dropped 2026-05-07). Currently scaffolding + Day 1 hello-world.

## Goal

Win the AMD Developer Hackathon (LabLab.ai, May 2026), Track 3, with a real-time webcam-based ASL → English speech translator. A deaf person signs → AI speaks. The demo IS the project: judges literally see two people who couldn't communicate, now do.

### Success criteria

- Submission accepted by 2026-05-11 03:00 MYT — live HF Space (Gradio) URL + 2–3 min demo video + lablab.ai submission form complete.
- End-to-end working flow: webcam frame → VLM recognizer → Llama-3.1-8B sentence composer → Coqui XTTS-v2 → speech output. **≤ 2 s** from capture to start of speech.
- V1 use cases: (1) ASL fingerspelling alphabet A–Z + 0–9, (2) Top-50 WLASL signs (hello, thank you, name, please, …). Target ≥ 75% accuracy on a 30-sample gold set.
- Reverse direction (speech → on-screen text for the deaf user) is a **stretch** for the buffer day only.
- Track 3: top-3 finish at minimum; gold target.

---

## Workflow tools

| Task | Skill / Plugin | Why |
|---|---|---|
| Planning (any non-trivial change) | `superpowers:writing-plans` | Hard rule — no free-form prose plans |
| Early-stage exploration | `superpowers:brainstorming` | Use before requirements firm |
| Executing the build plan | `superpowers:executing-plans` | Plan-driven implementation |
| Debugging | `superpowers:systematic-debugging` | Root-cause-first |
| Multi-agent / parallel sub-work | `superpowers:dispatching-parallel-agents` or `:subagent-driven-development` | Decompose by specialist |
| Pre-completion verification | `superpowers:verification-before-completion` | Don't claim done without checks |
| Test-driven implementation | `superpowers:test-driven-development` | Write test before code |
| Long-context cross-file analysis | `cc-gemini-plugin:gemini` | When 1M context window helps |
| Online docs lookup | `context7` (search/resolve) | "Verify online" rule — ROCm + HF + WLASL + MediaPipe specifics |
| Multi-source research with citations | `deep-research` | WLASL prior art, sign-language ML state of the art, ROCm performance |
| Whole-repo bug + logic audit | `deep-check` | 16-category systematic scan before submission |
| Second-opinion / rescue / stuck | `codex:rescue` | Hand off to Codex runtime |
| Code review (own work pre-submission) | `code-review:code-review` or `pr-review-toolkit:review-pr` | Style/bug/security pass before public release |
| Security review | `owasp-security` | OWASP Top 10 / ASVS — webcam + audio handling |
| Browser-based demo verification | `chrome-devtools-mcp:chrome-devtools` | Verify the HF Space before recording |
| Commit / push / PR | `commit-commands:commit-push-pr` | Standard commit flow |

**Hard rule:** every planning task goes through a `superpowers:*` skill — no free-form prose plans.

---

## Tech stack (locked)

- Languages: Python 3.12 (primary)
- Submission deliverable: Hugging Face Space (Gradio app, public, Apache 2.0)
- Inference backend: FastAPI on AMD Developer Cloud (single MI300X instance), exposed as OpenAI-compatible API
- Transport: HTTPS for V1; WebSocket only if latency demands it post-Day-2
- Pipeline (concurrent on one MI300X):
  - **Pose extraction:** MediaPipe Holistic (Google) — frame → 543-dim landmark vector
  - **Sign classifier:** trained-from-scratch small transformer over landmark sequences (WLASL Top-100 + ASL fingerspelling alphabet) → sign tokens
  - **Sentence composer:** `meta-llama/Llama-3.1-8B-Instruct` → grammatical English sentence from sign-token stream
  - **TTS:** `coqui/XTTS-v2` → audio
  - **(Stretch) STT:** `openai/whisper-large-v3` → reverse direction (speech → on-screen text)
- Datasets: [WLASL](https://github.com/dxli94/WLASL) Top-100 subset + ASL fingerspelling alphabet (open)
- HF Hub artifact: `lucas-loo/signbridge-classifier` (trained classifier weights + model card with ROCm training config)
- License: Apache 2.0
- GitHub mirror: https://github.com/seekerPrice/signbridge
- HF Space URL: https://huggingface.co/spaces/LucasLooTan/signbridge
- Submission link: *fill in once started on lablab.ai*

## Run Commands

```bash
# Setup (one-time)
pip install -r requirements.txt
cp .env.example .env  # fill in HF_TOKEN, AMD_DEV_CLOUD_*, OPENAI_API_KEY (fallback)

# Dev — run Gradio Space locally
python app.py

# Dev — run inference backend (locally for dev, deploys to AMD Dev Cloud for production)
python -m signbridge.backend

# Train the sign classifier on WLASL Top-100 (run on AMD Dev Cloud Day 2)
python -m signbridge.scripts.train_classifier --dataset data/wlasl --epochs 30

# Tests
pytest

# Lint / format / type
ruff check . && mypy signbridge/

# Push HF Space update (auto-deploys on git push to HF remote)
git push huggingface main
```

## Workspace layout

```
/Users/lucaslt/Documents/side-gig/amd-hackathon/
├── README.md                       # HF Space card via frontmatter
├── LICENSE                         # Apache 2.0
├── CLAUDE.md
├── .claude/
├── requirements.txt
├── .env.example
├── app.py                          # HF Space entry — Gradio
├── signbridge/
│   ├── __init__.py
│   ├── space.py                    # Gradio UI
│   ├── backend.py                  # FastAPI inference server
│   ├── recognizer/
│   │   ├── __init__.py
│   │   ├── landmarks.py            # MediaPipe Holistic wrapper
│   │   └── classifier.py           # trained sign classifier
│   ├── composer/
│   │   ├── __init__.py
│   │   └── sentence.py             # Llama-3.1-8B sentence composer
│   ├── voice/
│   │   ├── __init__.py
│   │   └── tts.py                  # Coqui XTTS-v2
│   └── scripts/
│       ├── __init__.py
│       └── train_classifier.py     # WLASL training script
├── data/
│   └── wlasl/                      # gitignored — WLASL Top-100 dataset
├── tests/
│   └── golden/                     # 30-sample gold set (Top-50 + alphabet)
└── docs/
    └── walkthrough.md              # technical walkthrough for submission
```

## References

- **Owner:** Lucas
- **Working dir:** `/Users/lucaslt/Documents/side-gig/amd-hackathon/`
- **Hackathon page:** https://lablab.ai/ai-hackathons/amd-developer
- **AMD article:** https://www.amd.com/en/developer/resources/technical-articles/2026/build-across-the-ai-stack--join-the-amd-x-lablab-ai-hackathon-.html
- **Track:** 3 (Vision & Multimodal AI). Extra Challenge (Build in Public) intentionally skipped 2026-05-07.
- **WLASL dataset:** https://github.com/dxli94/WLASL
- **MediaPipe Holistic:** https://developers.google.com/mediapipe/solutions/vision/holistic_landmarker
- **HF Space:** https://huggingface.co/spaces/LucasLooTan/signbridge (deployed 2026-05-07)
- **GitHub mirror:** https://github.com/seekerPrice/signbridge (deployed 2026-05-07)
- **Submission link:** *fill in once started on lablab.ai*
- **Plan file:** `/Users/lucaslt/.claude/plans/first-need-to-change-sparkling-dawn.md`

---

## Progress log (newest first)

**2026-05-07 — GitHub repo + HF Space live.** GitHub: `seekerPrice/signbridge`. HF Space: `LucasLooTan/signbridge` (Gradio SDK 4.44.1, Apache 2.0). All 16 source files mirrored to both. Awaiting AMD Dev Cloud credit email to wire up real VLM endpoint.

**2026-05-07 — Dropped Build-in-Public extra challenge.** Track 3 only. Frees ~2 hours that were earmarked for the 2 social posts + the external-facing walkthrough framing. Walkthrough doc kept as an internal technical record but no longer a submission deliverable.

**2026-05-07 — Pivoted to SignBridge.** Re-scored against the four judging criteria: SignBridge wins on Originality (10) and Presentation (10) thanks to the live deaf-person-to-hearing-person demo. Business value also stronger (Sorenson VRS comparable, mandated interpreter budgets). Replaced Iris scaffold (`iris/` package, README, requirements deps) with `signbridge/` package. CLAUDE.md, plan file, README rewritten. Day 1 hello-world starts: MediaPipe Holistic on webcam, WLASL data download, Plan-B VLM test.

**2026-05-07 — Initial Iris scaffold (deprecated).** Bootstrapped repo with Iris (visually-impaired navigation) plan, requirements.txt, .gitignore, .env.example, README. Replaced same-day after re-evaluation; kept reusable pieces (.gitignore, structural choices).
