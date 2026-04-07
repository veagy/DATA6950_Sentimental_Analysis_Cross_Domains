# Thesis documentation (IEEE LaTeX build)

All documentation for building the IEEE-style LaTeX report for this capstone lives in this folder.

| Document | Purpose |
|----------|---------|
| [LATEX_IEEE_BUILD_GUIDE.md](LATEX_IEEE_BUILD_GUIDE.md) | Prerequisites, toolchain, folder layout, citations, maintenance |
| [SOURCE_MAP.md](SOURCE_MAP.md) | Quick map: which `output/` paths feed which report sections |
| [FIGURE_SELECTION.md](FIGURE_SELECTION.md) | Curated figures copied into `latex/tables and visualizations/figures/` |
| [PAPER_SOURCE_INDEX.md](PAPER_SOURCE_INDEX.md) | Section-by-section map to repo files and metrics |

**LaTeX sources:** `../latex/main.tex` and `../latex/sections/`  
**Narrative source:** `../markdown/THESIS_IMPLEMENTATION.md`  
**Bibliography:** `../latex/references.bib`  

**Regenerate HRAST summary table for LaTeX:** `python ../scripts/csv_to_hrast_summary_tex.py` (from repo root).
