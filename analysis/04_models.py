import os, json, math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

R = os.environ.get("RESULTS_DIR", "results")
df = pd.read_parquet(os.path.join(R, "sample_metrics.parquet"))
counts = pd.read_csv(os.path.join(R, "counts.csv"))
sem = pd.read_csv(os.path.join(R, "semantic_metrics.csv"))
top = pd.read_csv(os.path.join(R, "topic_metrics.csv"))
ph = pd.read_csv(os.path.join(R, "phrase_counts.csv"))
auth = pd.read_csv(os.path.join(R, "author_panel.csv"))
within = pd.read_csv(os.path.join(R, "within_topic.csv"))

# lexical field-year aggregates
agg = df.groupby(["field","year"]).agg(
    n_lex=("paper_id","count"),
    mattr=("mattr","mean"),
    mattr_se=("mattr", lambda x: x.std(ddof=1)/math.sqrt(max(1,len(x)))),
    ttr=("ttr","mean"),
    compression=("compression","mean"),
    distinct2=("distinct2","mean"),
    entropy=("entropy","mean"),
    n_words=("n_words","mean"),
    n_sentences=("n_sentences","mean"),
    avg_sentence_len=("avg_sentence_len","mean"),
    llm_phrase_hits=("llm_phrase_hits","mean"),
    title_mattr=("title_mattr","mean"),
).reset_index()

panel = counts.merge(agg, on=["field","year"], how="left")
panel = panel.merge(sem, on=["field","year"], how="left")
panel = panel.merge(top, on=["field","year"], how="left")
panel = panel.merge(within, on=["field","year"], how="left")

# phrase composite: distinctive phrases only
distinct = ["delve","pivotal","intricate","realm","testament","underscore","tapestry",
            "seamless","holistic","meticulous","multifaceted","nuanced","notably",
            "crucial","landscape","navigate","leverage","paradigm","foster","showcase",
            "shed_light","crucial_role","important_note","realm_of","testament_to",
            "forefront","cutting_edge","game_changer","revolutionize","unprecedented",
            "transformative","pivotal_role","intricate_interplay","rich_tapestry",
            "delve_into","underscore_importance","navigate_complexities",
            "seamless_integration","holistic_approach"]
for c in distinct:
    if c in ph.columns:
        ph[c + "_rate"] = 1000 * ph[c] / ph["n"]
rate_cols = [c + "_rate" for c in distinct if c + "_rate" in ph.columns]
ph["llm_style_rate"] = ph[rate_cols].sum(axis=1)
panel = panel.merge(ph[["field","year","llm_style_rate"] + rate_cols], on=["field","year"], how="left")

AI_FIELDS = ["NLP","AI","ML"]
panel["treat"] = panel["field"].isin(AI_FIELDS).astype(int)
panel["post"] = (panel["year"] >= 2023).astype(int)
panel["treat_post"] = panel["treat"] * panel["post"]

panel.to_csv(os.path.join(R, "panel.csv"), index=False)

# ---------- descriptive pre/post ----------
pre = panel[panel.year.between(2017, 2022)]
post = panel[panel.year.between(2023, 2025)]
metrics = ["mattr","ttr","compression","distinct2","entropy","n_words","n_sentences",
           "avg_sentence_len","llm_phrase_hits","mean_cosine","within_topic_cosine",
           "vendi","cluster_entropy","hhi","topic_entropy","topic_hhi","llm_style_rate","title_mattr"]
desc_rows = []
for m in metrics:
    for grp, label in [(AI_FIELDS,"AI"), (["MATH","PHYSICS"],"Control")]:
        a = pre[pre.field.isin(grp)][m].mean()
        b = post[post.field.isin(grp)][m].mean()
        desc_rows.append((m,label,a,b,b-a, (b-a)/a if a and not pd.isna(a) else np.nan))
desc = pd.DataFrame(desc_rows, columns=["metric","group","pre_2017_2022","post_2023_2025","change","pct_change"])
desc.to_csv(os.path.join(R, "table2_prepost.csv"), index=False)

# ---------- DID regressions ----------
did_rows = []
for m in metrics:
    sub = panel.dropna(subset=[m]).copy()
    if len(sub) < 20:
        continue
    try:
        mod = smf.wls(f"{m} ~ C(field) + C(year) + treat:post", data=sub, weights=sub["n"]).fit(cov_type="HC1")
        beta = mod.params.get("treat:post", np.nan)
        se = mod.bse.get("treat:post", np.nan)
        p = mod.pvalues.get("treat:post", np.nan)
        did_rows.append((m, beta, se, p, beta-1.96*se, beta+1.96*se))
    except Exception as e:
        did_rows.append((m, np.nan, np.nan, np.nan, np.nan, np.nan))
did = pd.DataFrame(did_rows, columns=["metric","did","se","p","ci_lo","ci_hi"])
# Benjamini-Hochberg q-values across the pre-registered outcome family
from statsmodels.stats.multitest import multipletests
ok = did["p"].notna()
did["q_bh"] = np.nan
if ok.any():
    did.loc[ok, "q_bh"] = multipletests(did.loc[ok, "p"], method="fdr_bh")[1]
did.to_csv(os.path.join(R, "table3_did.csv"), index=False)

# leave-one-field-out robustness for the core semantic outcomes
loo_rows = []
for m in ["mean_cosine","within_topic_cosine","vendi"]:
    base = panel.dropna(subset=[m])
    estimates = []
    for drop in ["NLP","AI","ML","MATH","PHYSICS","CV","ROBOTICS","SYSTEMS"]:
        sub = base[base.field != drop].copy()
        if sub.treat.nunique() < 2:
            continue
        try:
            mod = smf.wls(f"{m} ~ C(field) + C(year) + treat:post", data=sub, weights=sub["n"]).fit(cov_type="HC1")
            estimates.append((drop, mod.params.get("treat:post"), mod.pvalues.get("treat:post")))
        except Exception:
            pass
    if estimates:
        vals = [e[1] for e in estimates]
        loo_rows.append((m, min(vals), max(vals), np.mean(vals), len(vals)))
loo = pd.DataFrame(loo_rows, columns=["metric","min_did","max_did","mean_did","n_specs"])
loo.to_csv(os.path.join(R, "table3e_leave_one_out.csv"), index=False)

# pre-trend test: treat-specific linear trend in the pre-period
pre_panel = panel[panel.year <= 2022].copy()
pre_panel["year_c"] = pre_panel["year"] - 2017
pretrend_rows = []
for m in ["mean_cosine","within_topic_cosine","vendi","cluster_entropy"]:
    sub = pre_panel.dropna(subset=[m])
    try:
        mod = smf.wls(f"{m} ~ C(field) + C(year) + treat:year_c", data=sub, weights=sub["n"]).fit(cov_type="HC1")
        pretrend_rows.append((m, mod.params.get("treat:year_c"), mod.bse.get("treat:year_c"), mod.pvalues.get("treat:year_c")))
    except Exception:
        pretrend_rows.append((m, np.nan, np.nan, np.nan))
pretrend = pd.DataFrame(pretrend_rows, columns=["metric","pretrend_coef","se","p"])
pretrend.to_csv(os.path.join(R, "table3f_pretrends.csv"), index=False)

# length-controlled robustness for text metrics
len_rows = []
for m in ["mattr","compression","title_mattr","llm_phrase_hits","mean_cosine","within_topic_cosine","vendi","cluster_entropy"]:
    sub = panel.dropna(subset=[m,"n_words","n_sentences"]).copy()
    try:
        mod = smf.wls(f"{m} ~ C(field) + C(year) + treat:post + n_words + n_sentences",
                      data=sub, weights=sub["n"]).fit(cov_type="HC1")
        len_rows.append((m, mod.params.get("treat:post", np.nan), mod.bse.get("treat:post", np.nan),
                         mod.pvalues.get("treat:post", np.nan)))
    except Exception:
        len_rows.append((m, np.nan, np.nan, np.nan))
len_ctrl = pd.DataFrame(len_rows, columns=["metric","did_len_controlled","se","p"])
len_ctrl.to_csv(os.path.join(R, "table3b_length_controlled.csv"), index=False)

# field-specific linear trends (more demanding specification)
trend_rows = []
panel["year_c"] = panel["year"] - 2022
for m in ["mattr","mean_cosine","within_topic_cosine","vendi","cluster_entropy","topic_hhi","llm_style_rate"]:
    sub = panel.dropna(subset=[m]).copy()
    try:
        mod = smf.wls(f"{m} ~ C(field) + C(year) + treat:post + C(field):year_c",
                      data=sub, weights=sub["n"]).fit(cov_type="HC1")
        trend_rows.append((m, mod.params.get("treat:post", np.nan), mod.bse.get("treat:post", np.nan),
                           mod.pvalues.get("treat:post", np.nan)))
    except Exception:
        trend_rows.append((m, np.nan, np.nan, np.nan))
trends = pd.DataFrame(trend_rows, columns=["metric","did_with_field_trends","se","p"])
trends.to_csv(os.path.join(R, "table3c_trends.csv"), index=False)

# ---------- event study ----------
event_rows = []
for m in ["mattr","mean_cosine","cluster_entropy","topic_hhi","llm_style_rate","llm_phrase_hits"]:
    for field in panel["field"].unique():
        sub = panel[(panel.field==field)].dropna(subset=[m]).copy()
        if len(sub) < 10:
            continue
        try:
            mod = smf.wls(f"{m} ~ C(year)", data=sub, weights=sub["n"]).fit()
            base = 2022
            for y in [2019,2020,2021,2022,2023,2024,2025]:
                key = f"C(year)[T.{y}]"
                if y == base:
                    event_rows.append((m, field, y, 0.0, 0.0))
                elif key in mod.params:
                    event_rows.append((m, field, y, mod.params[key], 0.0))
        except Exception:
            pass
ev = pd.DataFrame(event_rows, columns=["metric","field","year","coef","se"])
ev.to_csv(os.path.join(R, "table4_event_study.csv"), index=False)

# ---------- placebo: fake 2018 treatment, pre-period only ----------
placebo = panel[panel.year <= 2022].copy()
placebo["post_fake"] = (placebo["year"] >= 2018).astype(int)
placebo["treat_post_fake"] = placebo["treat"] * placebo["post_fake"]
plac_rows = []
for m in ["mattr","mean_cosine","within_topic_cosine","vendi","cluster_entropy","topic_hhi","llm_style_rate"]:
    sub = placebo.dropna(subset=[m])
    try:
        mod = smf.wls(f"{m} ~ C(field) + C(year) + treat:post_fake", data=sub, weights=sub["n"]).fit(cov_type="HC1")
        plac_rows.append((m, mod.params.get("treat:post_fake", np.nan), mod.bse.get("treat:post_fake", np.nan), mod.pvalues.get("treat:post_fake", np.nan)))
    except Exception:
        plac_rows.append((m, np.nan, np.nan, np.nan))
plac = pd.DataFrame(plac_rows, columns=["metric","did_fake","se","p"])
plac.to_csv(os.path.join(R, "table5_placebo.csv"), index=False)

# ---------- author panel ----------
inst_words = ["university", "institute", "academy", "college", "laboratory", "lab",
              "department", "school", "center", "centre", "research", "inc", "ltd",
              "corporation", "company", "hospital", "foundation"]
auth = auth[~auth["author"].str.contains("|".join(inst_words), case=False, na=False)].copy()
auth = auth[(auth["n_pre"] >= 3) & (auth["n_post"] >= 3)].copy()
for c in ["d_sem","d_mattr","d_ttr"]:
    lo, hi = auth[c].quantile([0.01, 0.99])
    auth[c] = auth[c].clip(lo, hi)
auth["treat"] = auth["field"].isin(AI_FIELDS).astype(int)
auth_rows = []
for m in ["d_sem","d_mattr","d_ttr"]:
    sub = auth.dropna(subset=[m])
    try:
        mod = smf.wls(f"{m} ~ treat", data=sub, weights=np.ones(len(sub))).fit(cov_type="HC1")
        auth_rows.append((m, mod.params.get("treat", np.nan), mod.bse.get("treat", np.nan), mod.pvalues.get("treat", np.nan), len(sub)))
    except Exception:
        auth_rows.append((m, np.nan, np.nan, np.nan, len(sub)))
auth_reg = pd.DataFrame(auth_rows, columns=["metric","ai_minus_control","se","p","n"])
auth_reg.to_csv(os.path.join(R, "table6_authors.csv"), index=False)
auth.groupby("field")[["d_sem","d_mattr","d_ttr"]].agg(["mean","count"]).to_csv(os.path.join(R, "table6_author_means.csv"))

# stricter author control: AI fields vs mathematics/physics only
strict = auth[auth.field.isin(["NLP","AI","ML","MATH","PHYSICS"])].copy()
strict["treat"] = strict["field"].isin(AI_FIELDS).astype(int)
strict_rows = []
for m in ["d_sem","d_mattr","d_ttr"]:
    sub = strict.dropna(subset=[m])
    try:
        mod = smf.wls(f"{m} ~ treat", data=sub, weights=np.ones(len(sub))).fit(cov_type="HC1")
        strict_rows.append((m, mod.params.get("treat"), mod.bse.get("treat"), mod.pvalues.get("treat"), len(sub)))
    except Exception:
        strict_rows.append((m, np.nan, np.nan, np.nan, len(sub)))
strict_reg = pd.DataFrame(strict_rows, columns=["metric","ai_minus_control","se","p","n"])
strict_reg.to_csv(os.path.join(R, "table6b_authors_strict.csv"), index=False)

# ---------- phrase ranking ----------
phrase_cols = [c for c in ph.columns if c.endswith("_rate") and c != "llm_style_rate"]
phrase_pre = ph[ph.year.between(2017,2022)][phrase_cols].mean()
phrase_post = ph[ph.year.between(2023,2025)][phrase_cols].mean()
phrase_change = (phrase_post - phrase_pre).sort_values(ascending=False)
phrase_change.to_csv(os.path.join(R, "table7_phrase_change.csv"))

# ---------- key stats json ----------
def get_did(m):
    row = did[did.metric==m]
    return None if row.empty else {k: (None if pd.isna(row.iloc[0][k]) else float(row.iloc[0][k])) for k in ["did","se","p","ci_lo","ci_hi"]}

key = {
    "n_papers_filtered": int(counts.n.sum()),
    "n_sample": int(len(df)),
    "fields": counts.groupby("field").n.sum().sort_values(ascending=False).to_dict(),
    "years": [int(counts.year.min()), int(counts.year.max())],
    "prepost": desc.to_dict(orient="records"),
    "did": {m: get_did(m) for m in ["mattr","ttr","compression","mean_cosine","within_topic_cosine","vendi","cluster_entropy","topic_hhi","llm_style_rate","llm_phrase_hits","n_words","title_mattr"]},
    "placebo": plac.to_dict(orient="records"),
    "authors": auth_reg.to_dict(orient="records"),
    "authors_strict": strict_reg.to_dict(orient="records"),
    "leave_one_out": loo.to_dict(orient="records"),
    "pretrends": pretrend.to_dict(orient="records"),
    "did_q": {m: (None if pd.isna(did[did.metric==m]["q_bh"].iloc[0]) else float(did[did.metric==m]["q_bh"].iloc[0])) for m in did.metric},
    "top_phrase_changes": {k: float(v) for k,v in phrase_change.head(15).items()},
}
with open(os.path.join(R, "key_stats.json"), "w") as f:
    json.dump(key, f, indent=2)

print("DID")
print(did.to_string(index=False))
print("\nPLACEBO")
print(plac.to_string(index=False))
print("\nAUTHORS")
print(auth_reg.to_string(index=False))
print("\nTOP PHRASE CHANGES (per 1000 abstracts, post minus pre)")
print(phrase_change.head(15).to_string())
print("\nDONE")
