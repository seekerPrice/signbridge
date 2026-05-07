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
    # Bind 0.0.0.0 by default — HF Spaces' reverse proxy expects the app to
    # listen on the container's external interface, and 127.0.0.1-only would
    # silently reject all incoming requests on the live Space. For local dev
    # boot-tests inside sandboxes that can't talk to 0.0.0.0, override with
    # `SIGNBRIDGE_HOST=127.0.0.1`.
    host = os.getenv("SIGNBRIDGE_HOST", "0.0.0.0")
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
