"""Regenerate docs/banner.png from docs/banner.html.

Usage:
    pip install playwright && playwright install chromium
    python docs/generate_banner.py

Edits to the banner: change docs/banner.html, re-run this script, commit both.
"""

from __future__ import annotations

import pathlib
import sys


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed. Run: pip install playwright && playwright install chromium")
        return 1

    here = pathlib.Path(__file__).resolve().parent
    html = (here / "banner.html").as_uri()
    out = here / "banner.png"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1400, "height": 720},
            device_scale_factor=2,  # retina-sharp output
        )
        page.goto(html)
        page.wait_for_load_state("networkidle")
        page.locator(".banner").screenshot(path=str(out))
        browser.close()

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
