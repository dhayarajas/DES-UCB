---
name: des-ucb-notebook-ui-testing
description: Run the DES-UCB notebook through JupyterLab and verify fresh results, inline plots, and honest claim reporting.
---

# Browser notebook testing

Run from the repository root using the existing `.venv`. The requirements include Jupyter.

```bash
.venv/bin/jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --IdentityProvider.token='' --ServerApp.password=''
```

Keep an unauthenticated server bound to loopback only. Open
`http://127.0.0.1:8888/lab/tree/DES_UCB.ipynb`.

Use **Run → Restart Kernel and Run All Cells**, then confirm **Restart**.
Use the left **Table of Contents** icon to navigate sections; widening the sidebar
makes the numbered headings readable. Maximize the browser before recording.

Committed notebook outputs and result files already exist: record their mtimes
before starting and verify fresh writes after the UI run. Save using Ctrl+S once
the kernel is Idle, then inspect saved execution counts and error outputs.
The notebook currently has 33 cells, 18 of which are executable Python cells;
Markdown cells do not have execution counts.

Datasets load in section 3; inspect its output for skipped downloads. ML-100K,
ML-1M, and MIND can use `data/` caches. Optional datasets may need user paths.
The default run takes roughly five minutes on an eight-core machine; section 14
prints measured wall time. Do not reduce experiment configuration just to finish.

Inspect actual inline figures, not only saved PNG files. Section 2 renders theta
trajectories and section 12 renders experiment figures.

Section 13 prints claim verdicts and writes `results/claims.json`; section 14
prints and writes `results/results_skeleton.md`. `NOT SUPPORTED` is an honest
research verdict, not a Python execution error. Distinguish it from `NOT RUN`
(missing coverage). Compare the registry verdicts with both paper-gate lists.

## Devin Secrets Needed

None for local Jupyter and cached/public datasets.
