"""Hugging Face Space entry point.

HF Spaces auto-discovers `app.py` at the repo root and launches whatever
Gradio interface it builds. Keep this file thin — real UI lives in
`signbridge.space`.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from signbridge.space import build_demo  # noqa: E402


def main() -> None:
    load_dotenv()
    # Make gradio's `_check_localhost` pre-flight skip itself — on HF Spaces
    # Docker the loopback connect-back occasionally races the bind and trips
    # the "When localhost is not accessible" guard. Setting SYSTEM=spaces
    # mirrors what the gradio-SDK runtime sets and is the documented escape
    # hatch. share=True was used earlier as a backup but HF Spaces warns it
    # is unsupported; SYSTEM=spaces alone is now sufficient.
    os.environ.setdefault("SYSTEM", "spaces")
    demo = build_demo()
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )


if __name__ == "__main__":
    main()
