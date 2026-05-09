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
    # Make gradio's `_check_localhost` pre-flight skip itself — on HF Spaces
    # Docker the loopback connect-back occasionally races the bind and trips
    # the "When localhost is not accessible" guard. Setting SYSTEM=spaces
    # mirrors what the gradio-SDK runtime sets and is the documented escape
    # hatch.
    os.environ.setdefault("SYSTEM", "spaces")
    demo = build_demo()
    # Gradio 4.44.1's _check_localhost pre-flight tries to connect to
    # 127.0.0.1:7860 from inside the container and fails on HF's Docker
    # SDK seccomp. Setting share=True is the documented bypass that skips
    # the check; the FRP tunnel it would normally create is suppressed
    # because HF detects the Space environment and serves directly.
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
