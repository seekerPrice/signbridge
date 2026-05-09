"""Hugging Face Space entry point.

HF Spaces auto-discovers `app.py` at the repo root and launches whatever
Gradio interface it builds. Keep this file thin — real UI lives in
`signbridge.space`.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from signbridge.space import build_demo


def main() -> None:
    load_dotenv()
    demo = build_demo()
    # Docker-SDK Space: we own the runtime, bind explicitly. Env vars from
    # the Dockerfile set GRADIO_SERVER_NAME=0.0.0.0 / PORT=7860 already; the
    # explicit args here are belt-and-suspenders.
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
