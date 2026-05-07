"""Generate the project cover image (1280×640 PNG).

Uses Pillow for a deterministic render — same output on every run, easy
to re-generate if branding changes. Output: assets/cover.png.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 640
OUT = Path("assets/cover.png")

# Brand palette (matches README frontmatter: indigo → pink)
INDIGO = (76, 29, 149)   # tailwind indigo-800
PINK = (219, 39, 119)    # tailwind pink-600
WHITE = (255, 255, 255)
SOFT = (240, 240, 250)


def _gradient_background(width: int, height: int) -> Image.Image:
    """Diagonal indigo→pink gradient."""
    img = Image.new("RGB", (width, height), INDIGO)
    px = img.load()
    for y in range(height):
        for x in range(width):
            t = (x + y) / (width + height)  # 0 → 1 along diagonal
            r = int(INDIGO[0] + (PINK[0] - INDIGO[0]) * t)
            g = int(INDIGO[1] + (PINK[1] - INDIGO[1]) * t)
            b = int(INDIGO[2] + (PINK[2] - INDIGO[2]) * t)
            px[x, y] = (r, g, b)
    return img


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Try a few common Mac/Linux system fonts; fall back to default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = _gradient_background(WIDTH, HEIGHT)
    d = ImageDraw.Draw(img)

    # Hands emoji icon (large, top-left feel)
    icon = _load_font(220, bold=True)
    d.text((80, 60), "🤟", font=icon, embedded_color=True)

    # Title
    title = _load_font(96, bold=True)
    d.text((360, 130), "SignBridge", font=title, fill=WHITE)

    # Tagline
    tagline = _load_font(38)
    d.text(
        (360, 250),
        "Real-time ASL → English speech",
        font=tagline,
        fill=SOFT,
    )

    # Pipeline strip
    pipeline = _load_font(26)
    d.text(
        (360, 310),
        "Qwen-VL  ·  Llama-3.1-8B  ·  XTTS-v2  ·  on AMD MI300X",
        font=pipeline,
        fill=SOFT,
    )

    # Track badge bottom-left
    badge = _load_font(24, bold=True)
    badge_text = "AMD Developer Hackathon  ·  Track 3 — Vision & Multimodal AI"
    bbox = d.textbbox((0, 0), badge_text, font=badge)
    pad_x, pad_y = 24, 14
    badge_w = bbox[2] - bbox[0] + pad_x * 2
    badge_h = bbox[3] - bbox[1] + pad_y * 2
    bx, by = 80, HEIGHT - badge_h - 60
    d.rounded_rectangle(
        (bx, by, bx + badge_w, by + badge_h), radius=12, fill=(0, 0, 0, 90)
    )
    d.text((bx + pad_x, by + pad_y - 4), badge_text, font=badge, fill=WHITE)

    img.save(OUT, format="PNG", optimize=True)
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
