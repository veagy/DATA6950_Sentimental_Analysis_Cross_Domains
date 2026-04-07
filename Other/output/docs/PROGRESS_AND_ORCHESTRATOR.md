# Progress reports and orchestrator session

## Dataset analysis notes — [`docs/progress_reports/dataset_analysis_points.txt`](../../docs/progress_reports/dataset_analysis_points.txt)

- **Yelp Academic:** ~7M reviews; long text vs short social data; 5-star skew (44% five-star).
- **Financial PhraseBank:** ~4.8k sentences; neutral-heavy (59%).
- **Healthcare mix:** HRAST (~27k), PatientStatements (~2k), MedicalSentiment (~400); informal vs clinical register.

## Progress_Report_01.md — week of 2026-01-26

- Traditional ML vs basic DL/transformers analysis; STACK1 ensemble +3.5 F1 target met.
- IMDB vs Sentiment140 difficulty characterization.
- Four domain datasets identified (Financial PhraseBank, HRAST, Medical, Yelp).
- **Next:** Acquire/preprocess new data, transformer optimization (full FT/LoRA), cross-domain tests, HRM refinement.

## Progress_Report_01.txt — 2026-02-09

- Overlaps with week-1 themes; adds **preprocessing pipeline** (dummy/preprocess.py in text—may refer to earlier layout), UTF-8 normalization, binary vs ternary labels, model code updates for multi-label.
- **Next:** ML baselines (LR, Linear SVC, NB, RF), then transformers/HRM vs ML.

## Progress_Report_02.txt — 2026-02-17

- Large implementation narrative: normalization/pooling/conv/dropout layers in `src/models/deep_leaning/models.py`, attention library in `attention_units.py`, `LLMModule`, clustering in `dummy.py` (to be reorganized).
- **Next:** HRM construction with new attention units, benchmarking attention variants, MoE routing refinement.

## Progress_Report_03.txt — 2026-03-25

- **Done:** Training runs for thesis configs (transformer, CNN, RNN) via `train_single.py` / `train_all.py` / `train_stack.py`; ML baselines on transformed features; HRM training in progress; thesis draft except Results chapter; validation via `validate_all.py` and unit tests.
- **Next:** MoE with `train_moe.py` and `config/moe/`, finish HRM + evaluation, Results chapter, inference standardization.

## Binary exports

`docs/progress_reports/` also contains **.docx** and **.pdf** versions of reports (and a `~$*` Office lock file). Content is not duplicated here; see [SOURCE_MANIFEST.txt](SOURCE_MANIFEST.txt) for paths.

## Orchestrator session: PreprocessText

| Artifact | Summary |
|----------|---------|
| [`master_plan.md`](../../docs/tasks/orchestrator-sessions/PreprocessText/master_plan.md) | Three subtasks: architect plan, implement `preprocess.py`, execute—all marked completed. |
| [`01_preprocess_text.task.md`](../../docs/tasks/orchestrator-sessions/PreprocessText/pending/01_preprocess_text.task.md) | Objective: batch-read raw IMDB/Sentiment140/Yelp JSON, `clean_text`, write `data/processed` parquet; deliver `Code/thesis/data/preprocess.py`. |
| [`Orchestrator_Summary.md`](../../docs/tasks/orchestrator-sessions/PreprocessText/Orchestrator_Summary.md) | Completed: dynamic discovery of raw files, Parquet output, chunked Yelp JSON; note on long-running background jobs for multi-GB sources. |

**Paths in orchestrator docs** reference `d:\CAPSTONE\capstone-2\`; map to your local `TEMP` or clone root.
