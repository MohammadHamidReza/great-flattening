# Results Directory

Derived data and tables for *The Great Flattening*.

| File | Description |
|---|---|
| `counts.csv` | Papers per field-year in the filtered corpus. |
| `sample.parquet` | Stratified analysis sample (139,158 papers). |
| `sample_metrics.parquet` | Per-paper lexical metrics, topic assignment, and embedding index. |
| `embeddings.npy` | 128-dimensional TF-IDF/SVD embeddings for the sample (row order matches `sample_metrics.parquet`). |
| `semantic_metrics.csv` | Field-year mean cosine similarity, Vendi, cluster entropy, HHI. |
| `within_topic.csv` | Within-topic semantic similarity by field-year. |
| `topic_metrics.csv` | Field-year topic entropy and HHI (50 global clusters). |
| `phrase_counts.csv` | Counts of 43 LLM-characteristic phrases by field-year. |
| `author_panel.csv` | Within-author pre/post changes. |
| `panel.csv` | Merged field-year panel used in the models. |
| `semantic_truncated.csv`, `within_topic_truncated.csv`, `panel_truncated.csv` | Length-standardized replication (120-token abstracts). |
| `table2_prepost.csv` | Pre/post means (Table A2). |
| `table3_did.csv` | Baseline difference-in-differences (Table A3). |
| `table3b_length_controlled.csv` | Length-controlled estimates (robustness). |
| `table3c_trends.csv` | Field-specific trend estimates (Table A4). |
| `table3d_truncated.csv` | 120-token truncation estimates (Table A5). |
| `table4_event_study.csv` | Descriptive event-time coefficients (no SE; one observation per year). |
| `table5_placebo.csv` | Fake-treatment placebo. |
| `table6_authors.csv`, `table6_author_means.csv` | Within-author regressions and means (Table A6). |
| `table7_phrase_change.csv` | Largest phrase increases (Table A7). |
| `key_stats.json` | Headline estimates used by the manuscript generator. |

## Notes

- Field groups: NLP (`cs.CL`), AI (`cs.AI`), ML (`cs.LG`), CV (`cs.CV`), robotics (`cs.RO`), mathematics (`math.*`), physics (selected `hep-*`, `quant-ph`, `gr-qc`, `astro-ph.*`, `cond-mat.*`), systems (`cs.SE`, `cs.CR`, `cs.DC`, `cs.PL`).
- AI treatment group in the DID models: NLP + AI + ML.
- Control group: mathematics + physics.
- Post period: 2023-2025. The 2026 data are partial and used only descriptively.
- The `table4_event_study.csv` standard errors are zero by construction because the event study is estimated on field-year means with one observation per year; it is descriptive, not inferential.
