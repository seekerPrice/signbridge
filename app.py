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
    # On HF Spaces the SERVER_NAME env defaults to 0.0.0.0; for local dev we
    # honour SIGNBRIDGE_HOST (default 127.0.0.1) so the boot test isn't blocked
    # by sandbox/proxy localhost-accessibility checks.
    host = os.getenv("SIGNBRIDGE_HOST", "127.0.0.1")
    port = int(os.getenv("SIGNBRIDGE_PORT", "7860"))
    demo.launch(
        server_name=host,
        server_port=port,
        show_error=True,
        share=False,
        prevent_thread_lock=False,
    )


if __name__ == "__main__":
    main()
