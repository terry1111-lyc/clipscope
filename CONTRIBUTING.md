# Contributing

Install `pip install -e '.[dev]'`, run `pytest -q`, and check browser code with `node --check src/clipscope/static/app.js`. Tests must not silently download models. The analytical test encoder belongs in tests only.

For model changes, record the checkpoint, activation, preprocessing, dependency versions, actual device and weight hash; rebuild the index. For metrics changes, add an analytically calculable ranking test including multiple relevant items.

Keep experiment configurations fixed before inspecting held-out results. Report regressions and negative findings. Preserve complete rankings and paired comparisons, and never promote synthetic numbers to claims about natural images.

Do not commit credentials, downloaded weights, private images, runtime indexes or third-party datasets. Check licenses before publishing new examples. Describe exactly which checks ran; distinguish a supplied deployment configuration from a verified deployment.
