"""Inline results/figs/*.png into report.html as base64 data URIs.
Replaces src="figs/NAME.png" with the data URI. Writes report_final.html."""

import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(src="report.html", dst="report_final.html"):
    html = (ROOT / src).read_text()

    def repl(m):
        p = ROOT / "results/figs" / m.group(1)
        if not p.exists():
            print("MISSING FIG:", m.group(1))
            return m.group(0)
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'src="data:image/png;base64,{b64}"'

    html = re.sub(r'src="figs/([^"]+)"', repl, html)
    (ROOT / dst).write_text(html)
    print("wrote", dst, len(html) // 1024, "KB")


if __name__ == "__main__":
    main(*sys.argv[1:])
