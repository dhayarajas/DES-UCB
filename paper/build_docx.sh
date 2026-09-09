#!/usr/bin/env bash
# Export the manuscript as a single-column Word document (main_onecol.docx).
# Requires pandoc >= 2.11 (for --citeproc). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"

cat docx_shim.tex tables/macros.tex body.tex > .body_docx.tex
sed -i \
  -e 's/\\includegraphics\[[^]]*\]{\([^}]*\)}/\\includegraphics{\1}/' \
  -e 's/\\begin{adjustbox}{[^}]*}//' \
  -e 's/\\end{adjustbox}//' \
  .body_docx.tex

pandoc .body_docx.tex \
  --from latex --to docx \
  --resource-path=.:figures:tables \
  --citeproc --bibliography refs.bib --csl ieee.csl \
  --metadata title="Drift-Triggered Evidence Switching Suppresses Stale Priors in Cold-Start Interactive Recommendation" \
  --metadata author="Dhayanidhi Rajasekaran (Research Scholar); Rajalakshmi N.R. (Prof. Dr.), Department of Computer Science and Engineering, Vel Tech Rangarajan Dr Sagunthala R&D Institute of Science and Technology, Chennai, India" \
  --number-sections \
  -o main_onecol.docx
rm -f .body_docx.tex
echo "wrote paper/main_onecol.docx"
