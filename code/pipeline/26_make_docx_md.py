#!/usr/bin/env python3
"""26_make_docx_md.py — Build the temporary markdown for the submission DOCX (figures embedded in the Figures section).

Reads paper/manuscript.md, inserts the matching image reference (width=6.3in) after each
**Fig. X** caption paragraph, and writes paper/.docx_build/manuscript_submission.md.
Pure file operations, no external calls.
"""
import os
import re
from pathlib import Path

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
OUT_DIR = BASE / "paper" / ".docx_build"
OUT_DIR.mkdir(exist_ok=True)

FIGS = {
    "Fig. 1": "fig1_design.png",
    "Fig. 2": "fig2_ladder.png",
    "Fig. 3": "fig3_frameworks.png",
    "Fig. 4": "fig4_paired.png",
    "Fig. 5": "fig5_exposure.png",
    "Fig. 6": "fig6_shasta.png",
    "Fig. 7": "fig7_multidam.png",
    "Fig. 8": "fig8_mechanism.png",
}

src = (BASE / "paper" / "manuscript.md").read_text()
lines = src.splitlines()
out = []
in_figs = False
for ln in lines:
    out.append(ln)
    if ln.strip().startswith("## Figures"):
        in_figs = True
        continue
    if in_figs:
        m = re.match(r"\*\*(Fig\. \d)\*\*", ln.strip())
        if m:
            key = m.group(1)
            png = FIGS.get(key)
            if png:
                out.append("")
                # The DOCX builder resolves figure paths from the repository root.
                out.append(f"![](figures/{png}){{width=6.3in}}")

(OUT_DIR / "manuscript_submission.md").write_text("\n".join(out) + "\n")
n_img = sum(1 for l in out if l.startswith("!["))
print(f"wrote {OUT_DIR / 'manuscript_submission.md'} ({n_img} embedded figures)")
