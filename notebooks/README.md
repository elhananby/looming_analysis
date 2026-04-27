# Notebooks

Use `compare.ipynb`, `canton-s.ipynb`, and `empty-split.ipynb` as package-based examples.

Avoid copying package functions into notebooks. Import from `looming_analysis` instead so
notebook behavior stays aligned with the tested package code.

`analysis.ipynb` is a legacy exploratory notebook and may contain old copied helper
functions. Prefer the `looming-analysis` config runner for repeatable runs, and use
`examples/quickstart.py` or package-import notebooks for custom exploratory analysis.
