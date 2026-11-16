# The Great Flattening â€” Study Design

**Date:** 2026-11-16
**Status:** Completed draft
**Paper:** `paper/manuscript.pdf`

## Question

Did the 2022 LLM transition reduce the diversity of AI researchâ€”not just its vocabulary, but its meaning?

## Why this design

The obvious hypothesisâ€”LLMs make writing more lexically repetitiveâ€”is testable with simple word-diversity measures. I tested it and found the opposite: lexical diversity increased. The paper therefore separates **lexical** from **semantic** diversity and treats the lexical result as a foil. The core evidence is semantic: mean pairwise cosine similarity, within-topic similarity, and the effective number of semantic modes (Vendi).

## Data

- arXiv Complete Corpus metadata snapshot (3,148,796 papers, 1986â€“2026).
- Filtered corpus: 1,525,557 abstracts (2010â€“2026, 200â€“4,000 characters, no withdrawals/duplicates).
- Stratified sample: 139,158 papers (up to 1,200 per field-year) for expensive semantic measures.
- Fields: NLP (cs.CL), AI (cs.AI), ML (cs.LG), vision (cs.CV), robotics (cs.RO), mathematics (math.*), physics (hep-th, hep-ph, quant-ph, gr-qc, astro-ph.CO, astro-ph.GA, cond-mat.str-el, cond-mat.mes-hall), systems (cs.SE, cs.CR, cs.DC, cs.PL).
- Controls: mathematics and physics.

## Measures

- **Lexical:** MATTR (window 50), TTR, distinct-2, unigram entropy, zlib compression ratio, title MATTR.
- **Semantic:** TF-IDF (40k features, 1â€“2 grams), truncated SVD (128 dims), mean pairwise cosine similarity, Vendi score, cluster entropy, topic HHI.
- **Within-topic:** mean cosine similarity among papers in the same global topic cluster.
- **Stylistic:** 43 pre-registered LLM-characteristic phrases per 1,000 abstracts.
- **Authors:** within-author change in self-similarity and TTR for authors with â‰¥3 pre- and â‰¥3 post-2023 papers.

## Identification

1. Difference-in-differences: AI fields vs. mathematics/physics, pre-2017â€“2022 vs. post-2023â€“2025, weighted by papers, HC1 standard errors.
2. Field-specific linear trends.
3. Within-author comparisons.
4. Fake-treatment placebo (2018).
5. Length-standardized replication: truncate every abstract to 120 tokens, refit TF-IDF/SVD, recompute semantic metrics.

## Key findings

- Lexical diversity **increased** in AI fields (+0.020 MATTR) but also in controls (+0.014); the AI-specific DID is small and disappears under length controls.
- Mean semantic similarity rose by 17.5% in AI vs. 2.7% in controls; DID +0.024 [0.009, 0.039].
- Within-topic similarity rose by 10.1% vs. 0.7%; DID +0.024 [0.011, 0.037].
- Vendi fell by 14.1% vs. 1.1%; DID âˆ’3.37 [âˆ’5.40, âˆ’1.34].
- Semantic estimates survive field-specific trends and 120-token truncation.
- LLM-characteristic phrases rose by 125 per 1,000 abstracts; placebo +58, reflecting a secular trend.
- Within-author self-similarity rose by 0.012 (p = 0.0004); TTR fell by 0.004 (p = 0.001).

## What is robust vs. suggestive

- **Robust:** mean semantic similarity, within-topic similarity, Vendi, author self-similarity.
- **Suggestive:** cluster entropy (not robust to truncation), topic HHI (not significant; placebo pre-trend), title diversity (not robust to length controls), LLM phrase index (secular trend).
- **Not supported:** lexical homogenization; topic concentration as a standalone claim.

## Limitations

Observational; arXiv-only; abstracts only; phrase list is a proxy; 2026 partial; semantic similarity â‰  conceptual diversity.

## Deliverables

- 13-page manuscript PDF and editable DOCX.
- Reproducible Python pipeline.
- Derived metrics and tables.
- 300 dpi figures.
- README, requirements, license, citation metadata, submission notes.

