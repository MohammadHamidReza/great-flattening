# Submission Notes

## What is in this package

- `paper/manuscript.pdf` — complete 13-page manuscript, with 7 figures and 10 appendix tables.
- `paper/manuscript.docx` — editable version of the same manuscript.
- `analysis/*.py` — full, reproducible pipeline (corpus build, metrics, models, figures, truncation replication).
- `results/*.csv` — derived metrics, panel data, tables, and headline estimates.
- `figures/*` — publication figures (PNG, 300 dpi).
- `README.md` — data provenance and reproduction instructions.
- `requirements.txt` — exact package versions.

## Before submitting

1. **Author metadata.** The manuscript uses "Mohammad Hamid Reza" / "Independent Researcher" / `mohammadhamidreza7@gmail.com`. Confirm how arXiv parses the given and family names.
2. **Repository.** The data-availability statement points to `https://github.com/MohammadHamidReza/great-flattening`. Confirm the repository is public.
3. **References.** The reference list is alphabetical and every entry is cited in the text. Do a final visual check before submission.
4. **Venue formatting.** The manuscript is a general preprint, not formatted for a specific venue.
5. **Claims.** The paper is observational. The title "More Words, Fewer Ideas" is a provocation; the measured construct is **semantic modes**, not ideas. Keep the hedged language.
6. **2026 data.** The snapshot ends in August 2026. The paper emphasizes 2023–2025.
7. **Ethics / disclosure.** No human subjects. Public metadata. Add a disclosure if the target venue requires one for AI-assisted writing.
8. **Placebo nuance.** The placebo is null for mean semantic similarity and Vendi, but not for LLM phrase rate or topic HHI. Do not remove those caveats.
9. **Length controls.** Adding abstract length attenuates the semantic estimates, but truncating all abstracts to 120 tokens preserves them. The paper reports both.

## Suggested venues

| Venue | Fit | Notes |
|---|---|---|
| **arXiv (cs.CY / cs.DL)** | Immediate preprint | Post first to establish priority. |
| **FAccT** | Strong | AI and society, measurement, science policy. |
| **NeurIPS Datasets & Benchmarks** | Strong if the dataset/dashboard is the main contribution | Emphasize the open measurement infrastructure. |
| **TMLR** | Strong | Rolling review, independent-friendly, rigorous empirical work. |
| **Nature Human Behaviour / PNAS** | High upside | Requires a stronger causal design and broader field coverage. |
| **Scientometrics** | Safe | Science-of-science audience; lower visibility. |

## Cover letter

> Dear Editors,
>
> I submit "The Great Flattening: More Words, Fewer Ideas in AI Research" for consideration. The paper provides the first large-scale measurement of how the 2022 LLM transition changed the diversity of AI research. Using 1.53 million arXiv abstracts and a difference-in-differences design with mathematics and physics as controls, I show a paradox: AI abstracts became lexically richer and longer, but their meaning converged. Mean semantic similarity and the effective number of semantic modes moved toward homogeneity, and the estimates survive field-specific trends and a length-standardized replication on truncated abstracts. Within-author comparisons show that the same researchers moved toward the new equilibrium. The paper also reports where the evidence is weaker: the within-topic estimate is sensitive to pre-trends, and the lexical and topic-concentration results do not survive all robustness checks.
>
> The results have implications for science policy, peer review, and the evaluation of AI systems. The paper is independent research. All code and derived data are available at https://github.com/MohammadHamidReza/great-flattening. The author declares no competing interests.
>
> Sincerely,
>
> Mohammad Hamid Reza
> Independent Researcher
> mohammadhamidreza7@gmail.com

## Robustness notes

- **Multiple comparisons:** mean semantic similarity, within-topic similarity, Vendi, and LLM phrase rate survive Benjamini–Hochberg correction. Cluster entropy does not (q = 0.073) and is labeled suggestive.
- **Pre-trends:** mean similarity and Vendi have null pre-trends (p = 0.64 and 0.24). Within-topic similarity has a positive pre-trend (p = 0.005) and a marginal placebo (p = 0.049); the paper treats it as supportive, not a standalone claim.
- **Leave-one-field-out:** mean similarity, within-topic similarity, and Vendi preserve their signs across all eight specifications.
- **Within-author:** self-similarity survives a strict AI-vs-math/physics control group (0.014, p = 0.026); the TTR decline does not (p = 0.75), and the paper says so.
- **Length controls:** adding abstract length attenuates the semantic estimates, but truncating every abstract to 120 tokens preserves them.

## Honest assessment

This is a strong preprint/workshop paper with a clear empirical contribution and a reproducible pipeline. For a top-tier venue, the main gaps are: (1) a sharper causal identification strategy, (2) broader field coverage beyond arXiv, and (3) a human validation of the phrase-based proxy. The semantic results are the robust core; the lexical, topic-concentration, cluster-entropy, and within-topic claims are explicitly qualified.
