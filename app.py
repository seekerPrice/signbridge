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
    # On HF Spaces the gradio runtime sets GRADIO_SERVER_NAME / SERVER_PORT
    # in env and auto-launches; passing custom server_name/port can collide
    # with the pre-startup localhost check. Calling .launch() with no args
    # is the documented "just works" pattern for HF Spaces.
    demo.queue().launch()


if __name__ == "__main__":
    main()
