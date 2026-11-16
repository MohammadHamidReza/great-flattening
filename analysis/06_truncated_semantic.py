import os, re, math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import statsmodels.formula.api as smf

R = os.environ.get("RESULTS_DIR", "results")
df = pd.read_parquet(os.path.join(R, "sample_metrics.parquet"))
counts = pd.read_csv(os.path.join(R, "counts.csv"))
rng = np.random.default_rng(42)

def truncate(s, n=120):
    words = re.findall(r"[A-Za-z0-9]+", str(s))
    return " ".join(words[:n])

print("Truncating abstracts to 120 tokens...")
texts = [truncate(s) for s in df["abstract"].tolist()]
fit_idx = rng.choice(len(texts), size=min(100000, len(texts)), replace=False)
vec = TfidfVectorizer(stop_words="english", max_features=40000, ngram_range=(1,2),
                      min_df=2, max_df=0.5, sublinear_tf=True)
vec.fit([texts[i] for i in fit_idx])
X = vec.transform(texts)
print("sparse", X.shape, X.nnz)
svd = TruncatedSVD(n_components=128, random_state=42, n_iter=5)
E = normalize(svd.fit_transform(X).astype(np.float32))
np.save(os.path.join(R, "embeddings_truncated.npy"), E)
print("explained", svd.explained_variance_ratio_.sum())

def entropy(counts):
    total = sum(counts)
    if total <= 0: return 0.0
    p = np.array([c/total for c in counts if c > 0], dtype=float)
    return float(-(p*np.log2(p)).sum())

rows = []
for (field, year), g in tqdm(df.groupby(["field","year"]), desc="sem"):
    idx = g.index.to_numpy()
    if len(idx) < 30: continue
    take = rng.choice(idx, size=min(400, len(idx)), replace=False)
    V = E[take]; S = V @ V.T; n = len(take)
    off = float((S.sum() - np.trace(S)) / (n*(n-1)))
    K = S / n
    ev = np.clip(np.linalg.eigvalsh(K), 0, None); ev = ev/ev.sum()
    vendi = float(np.exp(-(ev[ev>0]*np.log(ev[ev>0])).sum()))
    k = min(20, max(2, n//20))
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(V)
    cnt = np.bincount(km.labels_, minlength=k)
    ent = entropy(cnt)/math.log(k)
    hhi = float(((cnt/n)**2).sum())
    rows.append((field,year,n,off,vendi,ent,hhi))
sem = pd.DataFrame(rows, columns=["field","year","n_sem","mean_cosine","vendi","cluster_entropy","hhi"])
sem.to_csv(os.path.join(R, "semantic_truncated.csv"), index=False)

# within-topic on truncated embeddings
wrows = []
for (field, year), g in tqdm(df.groupby(["field","year"]), desc="within"):
    sims, weights = [], []
    for t, tg in g.groupby("topic"):
        if len(tg) < 20: continue
        idx = tg.index.to_numpy(); take = rng.choice(idx, size=min(100, len(idx)), replace=False)
        V = E[take]; S = V @ V.T; n = len(take)
        sims.append(float((S.sum() - np.trace(S))/(n*(n-1)))); weights.append(len(tg))
    if sims:
        wrows.append((field,year,float(np.average(sims, weights=weights)),int(sum(weights))))
within = pd.DataFrame(wrows, columns=["field","year","within_topic_cosine","n_within"])
within.to_csv(os.path.join(R, "within_topic_truncated.csv"), index=False)

panel = counts.merge(sem, on=["field","year"], how="left").merge(within, on=["field","year"], how="left")
panel["treat"] = panel["field"].isin(["NLP","AI","ML"]).astype(int)
panel["post"] = (panel["year"] >= 2023).astype(int)
panel.to_csv(os.path.join(R, "panel_truncated.csv"), index=False)

rows = []
for m in ["mean_cosine","within_topic_cosine","vendi","cluster_entropy"]:
    sub = panel.dropna(subset=[m])
    try:
        mod = smf.wls(f"{m} ~ C(field) + C(year) + treat:post", data=sub, weights=sub["n"]).fit(cov_type="HC1")
        rows.append((m, mod.params.get("treat:post"), mod.bse.get("treat:post"), mod.pvalues.get("treat:post"),
                     mod.params.get("treat:post")-1.96*mod.bse.get("treat:post"),
                     mod.params.get("treat:post")+1.96*mod.bse.get("treat:post")))
    except Exception:
        rows.append((m,np.nan,np.nan,np.nan,np.nan,np.nan))
did = pd.DataFrame(rows, columns=["metric","did","se","p","ci_lo","ci_hi"])
did.to_csv(os.path.join(R, "table3d_truncated.csv"), index=False)
print(did.to_string(index=False))
print("DONE")
