import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

R = os.environ.get("RESULTS_DIR", "results")
F = os.environ.get("FIGURES_DIR", "figures")
os.makedirs(F, exist_ok=True)

panel = pd.read_csv(os.path.join(R, "panel.csv"))
ev = pd.read_csv(os.path.join(R, "table4_event_study.csv"))
auth = pd.read_csv(os.path.join(R, "author_panel.csv"))
did = pd.read_csv(os.path.join(R, "table3_did.csv"))
plac = pd.read_csv(os.path.join(R, "table5_placebo.csv"))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

COL = {"NLP":"#C0392B","AI":"#E67E22","ML":"#8E44AD","CV":"#27AE60",
       "MATH":"#2E86C1","PHYSICS":"#7F8C8D","SYSTEMS":"#16A085","ROBOTICS":"#A04000"}
LABEL = {"NLP":"NLP (cs.CL)","AI":"AI (cs.AI)","ML":"ML (cs.LG)","CV":"Vision (cs.CV)",
         "MATH":"Mathematics","PHYSICS":"Physics","SYSTEMS":"Systems","ROBOTICS":"Robotics"}
ORDER = ["NLP","AI","ML","CV","MATH","PHYSICS","SYSTEMS","ROBOTICS"]

def line(metric, title, ylabel, fname, fields=None, se=None, hline=None, log=False):
    fig, ax = plt.subplots(figsize=(7.2,4.2))
    fields = fields or [f for f in ORDER if f in panel.field.unique()]
    for f in fields:
        d = panel[panel.field==f].sort_values("year")
        d = d[d.year<=2026]
        ax.plot(d.year, d[metric], marker="o", ms=2.5, lw=1.6, color=COL.get(f,"k"), label=LABEL.get(f,f))
        if se and se in d.columns:
            ax.fill_between(d.year, d[metric]-1.96*d[se], d[metric]+1.96*d[se], color=COL.get(f,"k"), alpha=0.10, lw=0)
    ax.axvline(2022.85, color="#333", ls="--", lw=1, alpha=0.7)
    ax.text(2022.95, ax.get_ylim()[1]*0.98, "ChatGPT", fontsize=8, va="top")
    if hline is not None: ax.axhline(hline, color="k", lw=0.6, alpha=0.4)
    if log: ax.set_yscale("log")
    ax.set_title(title); ax.set_xlabel("Year"); ax.set_ylabel(ylabel)
    ax.legend(ncol=2, fontsize=7.5, frameon=False)
    fig.savefig(os.path.join(F, fname)); plt.close(fig)

line("n", "arXiv submissions in the analysis sample", "Papers per field-year", "fig1_corpus.png", log=True)
line("mattr", "Lexical diversity of abstracts", "MATTR (content words, window 50)", "fig2_lexical.png", se="mattr_se")
line("compression", "Repetitiveness of abstracts", "Compression ratio (lower = more repetitive)", "fig3_compression.png")
line("mean_cosine", "Semantic similarity between abstracts", "Mean pairwise cosine similarity", "fig4_semantic.png")
line("cluster_entropy", "Semantic cluster diversity", "Normalized cluster entropy", "fig5_cluster_entropy.png")
line("topic_hhi", "Topic concentration", "Topic HHI (higher = more concentrated)", "fig6_topic_hhi.png")
line("llm_style_rate", "LLM-characteristic phrase rate", "Phrase mentions per 1,000 abstracts", "fig7_phrases.png")
line("vendi", "Effective semantic diversity", "Vendi score (effective number of modes)", "fig11_vendi.png")
line("within_topic_cosine", "Within-topic semantic similarity", "Mean cosine similarity within topic", "fig12_within_topic.png")

# Event study: MATTR and semantic similarity
fig, axes = plt.subplots(1,2, figsize=(9.2,3.8))
for ax, m, title in [(axes[0],"mattr","Lexical diversity (MATTR)"),(axes[1],"mean_cosine","Semantic similarity")]:
    for f in ["NLP","ML","MATH","PHYSICS"]:
        d = ev[(ev.metric==m)&(ev.field==f)].sort_values("year")
        if d.empty: continue
        ax.errorbar(d.year, d.coef, yerr=1.96*d.se, marker="o", ms=2.5, lw=1.5,
                    capsize=2, color=COL.get(f,"k"), label=LABEL.get(f,f))
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title(title); ax.set_xlabel("Year (2022 = 0)"); ax.set_ylabel("Coefficient")
    ax.legend(fontsize=7, frameon=False)
fig.savefig(os.path.join(F,"fig8_event_study.png")); plt.close(fig)

# Author panel
fig, axes = plt.subplots(1,2, figsize=(8.2,3.6))
for ax, m, title in [(axes[0],"d_sem","Change in self-similarity"),(axes[1],"d_mattr","Change in lexical diversity")]:
    data = [auth[auth.field==f][m].dropna().values for f in ["NLP","ML","MATH","PHYSICS"]]
    ax.boxplot(data, tick_labels=["NLP","ML","Math","Physics"], showfliers=False, widths=0.55)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title(title); ax.set_ylabel("Post − pre")
fig.savefig(os.path.join(F,"fig9_author_panel.png")); plt.close(fig)

# DID vs placebo
fig, ax = plt.subplots(figsize=(7.2,3.6))
metrics = ["mattr","mean_cosine","cluster_entropy","topic_hhi","llm_style_rate"]
x = np.arange(len(metrics)); w=0.36
main = [did[did.metric==m]["did"].values[0] if not did[did.metric==m].empty else np.nan for m in metrics]
fake = [plac[plac.metric==m]["did_fake"].values[0] if not plac[plac.metric==m].empty else np.nan for m in metrics]
ax.bar(x-w/2, main, w, label="Main DID (2023+)", color="#C0392B")
ax.bar(x+w/2, fake, w, label="Placebo (2018)", color="#95A5A6")
ax.axhline(0, color="k", lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(["MATTR","Semantic sim.","Cluster entropy","Topic HHI","LLM phrases"], rotation=15, ha="right")
ax.set_ylabel("DID coefficient"); ax.set_title("Main effects vs. placebo")
ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(F,"fig10_did_placebo.png")); plt.close(fig)

print("figures written to", F)
for fn in sorted(os.listdir(F)):
    print(fn, os.path.getsize(os.path.join(F,fn)))
