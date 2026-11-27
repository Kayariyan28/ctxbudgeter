"""Regenerate docs/banner.png (2560x1280) from docs/banner/ — a K28 D2 page.

Usage:
    python docs/generate_banner.py            # needs Google Chrome or Chromium installed

The page is authored at 1920x960 (2:1) in the locked K28 D2 design system: edit
docs/banner/page.src.html, keep docs/banner/index.html in sync (it is the same page;
the design-systems skill's expand.py regenerates it), re-run this script, commit all.
Fonts are local woff2 files in docs/banner/assets/fonts, so renders are deterministic.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

CANDIDATES = [
    os.environ.get("CHROME", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
    shutil.which("chromium-browser") or "",
]


def main() -> int:
    chrome = next((c for c in CANDIDATES if c and os.path.exists(c)), None)
    if chrome is None:
        print("Chrome/Chromium not found. Install it or set CHROME=/path/to/chrome.")
        return 1
    here = pathlib.Path(__file__).resolve().parent
    page = (here / "banner" / "index.html").as_uri() + "#banner"
    out = here / "banner.png"
    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
         "--allow-file-access-from-files", "--force-device-scale-factor=1.3333334",
         "--window-size=1920,960", "--virtual-time-budget=4000",
         "--run-all-compositor-stages-before-draw", f"--screenshot={out}", page],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
