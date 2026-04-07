# LaTeX paper: source file index

Maps each compiled section in `output/thesis/latex/sections/` to repository inputs. Use this when updating numbers or figures.

**Source hierarchy**

- **Primary (structured):** `markdown/THESIS_IMPLEMENTATION.md` — section map I–IX aligned with the LaTeX outline; use for factual scope and cross-checks.
- **Secondary (verbatim thesis PDF extracts):** `THESIS_REPORT_NEW.md` and `THESIS_REPORT.md` — optional prose pulls for Introduction / Motivation / Methodology / Datasets **only** after de-hyphenating line breaks and replacing bracket cites `[n]` with `\cite{refNN}` when `refNN` exists in `latex/references.bib`. Do not duplicate tables already in `thesis_results.tex`, `datasets.tex`, or appendix fragments.
- **Tertiary (capstone planning / slides):** `docs/docs/THESIS_MASTER_DOCUMENT.md`, `docs/docs/presentation_slides.md`, `docs/docs/p1-p2-p3_merged.md` (strip `[cite_start]…` artifacts), `docs/docs/MODEL_*.md`, `PYTHON_MODEL_CLASSES.md`; **figures:** `output/thesis/imgs/new/*.png` (copied per `FIGURE_SELECTION.md`); **midterm deck:** `docs/CapStone-II MidTerm Presentation.pdf` for bullets not captured in markdown.

| LaTeX file | Primary sources | Secondary (optional prose) |
|------------|-----------------|----------------------------|
| `abstract.tex` | `THESIS_IMPLEMENTATION.md` (Abstract) | Master doc executive summary; headline metrics must match §Abstract |
| `introduction.tex` | `THESIS_IMPLEMENTATION.md` §I | `presentation_slides.md` (problem/thesis/objectives); `p1-p2-p3_merged.md` §1 cleaned |
| `related_work.tex` | `THESIS_IMPLEMENTATION.md` §II | Matching related-work passages in report markdown |
| `methodology.tex` | `THESIS_IMPLEMENTATION.md` §III | Methodology chapter in `THESIS_REPORT_NEW.md` for extra training/preprocessing detail |
| `datasets.tex` | `THESIS_IMPLEMENTATION.md` §IV, §IX-A, `output/dataset analysis/` | `tab:processed_stems` from `data/processed/*.parquet` stems (exclude `all-data`) |
| `experimental_setup.tex` | `THESIS_IMPLEMENTATION.md` §III-F, §VII-A | `presentation_slides.md` (methodology/milestones); midterm PDF |
| `capstone_visuals.tex` | — | `output/thesis/imgs/new` → `figures/fig_capstone_*.png` (see `FIGURE_SELECTION.md`); numeric disclaimers in-section |
| `thesis_results.tex` | `THESIS_IMPLEMENTATION.md` §V (tables V-A–V-F); EDA PNGs from `output/charts/dataset_analysis/` | Use report only to verify wording vs. tables |
| `results.tex` | `THESIS_IMPLEMENTATION.md` §IX, `output/metrics/summary.csv`, `output/charts/metrics/` | — |
| `discussion.tex` | `THESIS_IMPLEMENTATION.md` §VI | Midterm bar/scatter figures (qualitative only) |
| `conclusion.tex` | `THESIS_IMPLEMENTATION.md` §VIII | Master doc closing + midterm deliverables wording |
| `hrm_finetune_validation.tex` | `THESIS_IMPLEMENTATION.md` (HRM eval path); `validate_hrm_finetune_per_dataset.py` outputs | — |
| `system_artifacts.tex` | `output/models/index.md`, `output/path/summary_by_topdir.md` | — |
| `repository_snapshot.tex` | `THESIS_IMPLEMENTATION.md` §IX, `output/dataset analysis/alignment_report.md` | — |
| `appendix.tex` | `output/metrics/summary.csv` via `csv_to_hrast_summary_tex.py` → `tables/.../tab_hrast_summary.tex` | — |

**Automation:** `output/thesis/scripts/csv_to_hrast_summary_tex.py` regenerates the appendix HRAST table (drops rows without both `accuracy` and `f1_macro`; failed loads stay in `summary.csv`).

**Bibliography:** `latex/references.bib` only; citation keys `ref01`–`ref67`.
