# The Great Flattening

**Working paper:** *The Great Flattening: More Words, Fewer Ideas in AI Research*

**Author:** Mohammad Hamid Reza (Independent Researcher)

**Repository:** https://github.com/MohammadHamidReza/great-flattening

This repository contains the analysis code, derived metrics, figures, and manuscript for an independent study of how AI research changed after the 2022 LLM transition. The study uses public arXiv metadata (2010–2026) and compares AI fields with mathematics and physics.

## Headline finding

AI abstracts became lexically richer and longer after 2022, but their meaning converged. Mean pairwise semantic similarity and the effective number of semantic modes (Vendi) moved toward homogeneity in AI fields relative to controls. The semantic estimates survive field-specific trends and a length-standardized replication on 120-token abstracts. The pattern is strongest in NLP, weaker in ML/AI, and largely absent in mathematics and physics.

## Structure

```
analysis/
  00_schema.py                inspect the raw metadata
  01_inspect.py               quick metadata audit
  02_build_sample.py          filter corpus, build stratified sample, count phrases
  03_metrics.py               lexical, semantic, topic, and author-level metrics
  04_models.py                DID, field trends, placebo, author panel
  05_figures.py               figures
  06_truncated_semantic.py    length-standardized replication on 120-token abstracts
paper/
  manuscript.pdf              compiled 13-page manuscript
  manuscript.docx             editable manuscript
  abstract.txt                title, author, abstract, keywords
  arxiv_submission.md         arXiv metadata checklist
  cover_letter.md             cover letter for journal submissions
results/
  counts.csv                  papers per field-year
  semantic_metrics.csv        field-year semantic diversity
  semantic_truncated.csv      length-standardized semantic metrics
  topic_metrics.csv           field-year topic concentration
  within_topic.csv            within-topic semantic similarity
  phrase_counts.csv           LLM-characteristic phrase counts
  author_panel.csv            within-author changes
  panel.csv                   merged field-year panel
  table*.csv                  manuscript tables
  key_stats.json              headline estimates
figures/                      publication figures
docs/DESIGN.md                study design and key decisions
```

## Data

The raw data is the public **arXiv Complete Corpus** metadata snapshot (harvested August 2026), available at:

`https://huggingface.co/datasets/secemp9/arxiv-complete`

The metadata parquet is ~1.64 GB and contains 3,148,796 papers. I do not redistribute the raw metadata. The derived metrics in `results/` are released for reproducibility.

## Reproduce

Use Python 3.12 and the packages in `requirements.txt`. Set the environment variables:

```powershell
$env:ARXIV_PARQUET="data/arxiv_metadata.parquet"
$env:RESULTS_DIR="results"
python analysis/02_build_sample.py
python analysis/03_metrics.py
python analysis/04_models.py
python analysis/05_figures.py
python analysis/06_truncated_semantic.py
```

## Key caveats

- The study is observational. Estimates are associations consistent with a causal interpretation, not definitive causal effects.
- The 2026 data are partial (through August 2026).
- Phrase lists are proxies for LLM-assisted writing, not detectors.
- The title "More Words, Fewer Ideas" is a provocation; the measured construct is **semantic modes**, not ideas.
- The placebo is null for mean semantic similarity and Vendi, but not for LLM phrase rate or topic HHI; the paper reports these caveats.

## License

Code: MIT. Derived metrics: CC BY 4.0. Raw arXiv metadata remains subject to the original arXiv licenses.
