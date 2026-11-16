import os, re, zlib, math, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

RESULTS = os.environ.get("RESULTS_DIR", "results")
SAMPLE = os.environ.get("SAMPLE_PATH", os.path.join(RESULTS, "sample.parquet"))

print("Loading sample...")
df = pd.read_parquet(SAMPLE)
df = df.reset_index(drop=True)
print(len(df), "rows")

STOP = set(ENGLISH_STOP_WORDS)

LLM_PHRASES = [
    "delve", "pivotal", "intricate", "realm", "testament", "underscore", "tapestry",
    "seamless", "holistic", "meticulous", "multifaceted", "nuanced", "notably",
    "furthermore", "moreover", "crucial", "landscape", "navigate", "leverage",
    "robust", "comprehensive", "paradigm", "foster", "showcase", "shed light",
    "plays a crucial role", "it is important to note", "in the realm of",
    "a testament to", "at the forefront", "cutting-edge", "game-changer",
    "revolutionize", "unprecedented", "transformative", "pivotal role",
    "intricate interplay", "rich tapestry", "delve into", "underscore the importance",
    "navigate the complexities", "seamless integration", "holistic approach",
]

LATEX_RE = re.compile(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?")
MATH_RE = re.compile(r"\$[^$]*\$")
URL_RE = re.compile(r"https?://\S+")
CITE_RE = re.compile(r"\[\s*\d+(\s*,\s*\d+)*\s*\]")
NONALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-z]{2,}")
SENT_RE = re.compile(r"[.!?]+")

def clean_text(s):
    s = LATEX_RE.sub(" ", s)
    s = MATH_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = CITE_RE.sub(" ", s)
    s = NONALPHA_RE.sub(" ", s.lower())
    return SPACE_RE.sub(" ", s).strip()

def mattr(words, window=50):
    if len(words) < window:
        return len(set(words)) / max(1, len(words))
    vals = []
    step = max(1, window // 5)
    for i in range(0, len(words) - window + 1, step):
        w = words[i:i+window]
        vals.append(len(set(w)) / window)
    return float(np.mean(vals)) if vals else len(set(words)) / len(words)

def entropy(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    p = np.array([c / total for c in counts if c > 0], dtype=float)
    return float(-(p * np.log2(p)).sum())

rows = []
for r in tqdm(df.itertuples(index=False), total=len(df), desc="lexical"):
    raw = str(r.abstract)
    low = raw.lower()
    words_all = WORD_RE.findall(clean_text(raw))
    words = [w for w in words_all if w not in STOP]
    n = len(words)
    if n < 20:
        rows.append((r.paper_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        continue
    uniq = len(set(words))
    ttr = uniq / n
    bigrams = list(zip(words, words[1:]))
    dist2 = len(set(bigrams)) / max(1, len(bigrams))
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    ent = entropy(list(counts.values()))
    comp = len(zlib.compress(raw.encode("utf-8", errors="ignore"))) / max(1, len(raw.encode("utf-8", errors="ignore")))
    sents = [s for s in SENT_RE.split(raw) if len(s.strip()) > 5]
    sent_cnt = len(sents)
    sent_len = np.mean([len(WORD_RE.findall(s)) for s in sents]) if sents else 0
    phrase_hits = sum(low.count(p) for p in LLM_PHRASES)
    title_words = [w for w in WORD_RE.findall(clean_text(str(r.title))) if w not in STOP]
    t_mattr = mattr(title_words) if len(title_words) >= 8 else len(set(title_words)) / max(1, len(title_words))
    rows.append((r.paper_id, n, ttr, mattr(words), dist2, ent, comp, sent_cnt, sent_len, phrase_hits, len(words_all), t_mattr))

cols = ["paper_id","n_words","ttr","mattr","distinct2","entropy","compression","n_sentences","avg_sentence_len","llm_phrase_hits","n_words_all","title_mattr"]
lex = pd.DataFrame(rows, columns=cols)
df = df.merge(lex, on="paper_id", how="left")
print(df[["mattr","ttr","compression","llm_phrase_hits"]].describe().to_string())

# ---------- semantic embeddings ----------
print("\nFitting TF-IDF...")
texts = df["abstract"].astype(str).tolist()
rng = np.random.default_rng(42)
fit_idx = rng.choice(len(texts), size=min(100000, len(texts)), replace=False)
vec = TfidfVectorizer(stop_words="english", max_features=40000, ngram_range=(1,2),
                      min_df=2, max_df=0.5, sublinear_tf=True)
vec.fit([texts[i] for i in fit_idx])
print("vocab", len(vec.vocabulary_))

print("Transforming...")
X = vec.transform(texts)
print("sparse", X.shape, X.nnz)

print("SVD...")
svd = TruncatedSVD(n_components=128, random_state=42, n_iter=5)
E = svd.fit_transform(X).astype(np.float32)
E = normalize(E)
np.save(os.path.join(RESULTS, "embeddings.npy"), E)
print("explained", svd.explained_variance_ratio_.sum())

# ---------- semantic metrics per field-year ----------
print("\nSemantic metrics...")
sem_rows = []
for (field, year), g in tqdm(df.groupby(["field","year"]), desc="semantic"):
    idx = g.index.to_numpy()
    if len(idx) < 30:
        continue
    take = rng.choice(idx, size=min(400, len(idx)), replace=False)
    V = E[take]
    S = V @ V.T
    n = len(take)
    off = (S.sum() - np.trace(S)) / (n*(n-1))
    # Vendi score
    K = S / n
    ev = np.linalg.eigvalsh(K)
    ev = np.clip(ev, 0, None)
    ev = ev / ev.sum()
    vendi = float(np.exp(-(ev[ev>0] * np.log(ev[ev>0])).sum()))
    # cluster entropy + HHI
    k = min(20, max(2, n // 20))
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(V)
    counts = np.bincount(km.labels_, minlength=k)
    ent = entropy(counts) / math.log(k)
    hhi = float(((counts / n) ** 2).sum())
    sem_rows.append((field, year, n, float(off), vendi, float(ent), hhi))
sem = pd.DataFrame(sem_rows, columns=["field","year","n_sem","mean_cosine","vendi","cluster_entropy","hhi"])
sem.to_csv(os.path.join(RESULTS, "semantic_metrics.csv"), index=False)
print(sem.head().to_string())

# ---------- global topic concentration ----------
print("\nTopic metrics...")
topic_idx = rng.choice(len(E), size=min(50000, len(E)), replace=False)
km = KMeans(n_clusters=50, n_init=5, random_state=42).fit(E[topic_idx])
df["topic"] = km.predict(E)
topic_rows = []
for (field, year), g in tqdm(df.groupby(["field","year"]), desc="topics"):
    if len(g) < 30:
        continue
    counts = np.bincount(g["topic"].values, minlength=50)
    topic_rows.append((field, year, len(g), entropy(counts)/math.log(50), float(((counts/len(g))**2).sum())))
top = pd.DataFrame(topic_rows, columns=["field","year","n_topic","topic_entropy","topic_hhi"])
top.to_csv(os.path.join(RESULTS, "topic_metrics.csv"), index=False)
print(top.head().to_string())

# ---------- within-topic semantic similarity ----------
print("\nWithin-topic similarity...")
within_rows = []
for (field, year), g in tqdm(df.groupby(["field","year"]), desc="within-topic"):
    sims, weights = [], []
    for t, tg in g.groupby("topic"):
        if len(tg) < 20:
            continue
        idx = tg.index.to_numpy()
        take = rng.choice(idx, size=min(100, len(idx)), replace=False)
        V = E[take]
        S = V @ V.T
        n = len(take)
        sims.append(float((S.sum() - np.trace(S)) / (n*(n-1))))
        weights.append(len(tg))
    if sims:
        within_rows.append((field, year, float(np.average(sims, weights=weights)), int(sum(weights))))
within = pd.DataFrame(within_rows, columns=["field","year","within_topic_cosine","n_within"])
within.to_csv(os.path.join(RESULTS, "within_topic.csv"), index=False)
print(within.head().to_string())

# ---------- author panel ----------
print("\nAuthor panel...")
def parse_authors(s):
    s = str(s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"[^A-Za-z\s,;]", " ", s)
    parts = re.split(r"[,;]|\band\b", s)
    out = []
    for p in parts:
        p = SPACE_RE.sub(" ", p).strip().lower()
        if len(p) >= 5 and len(p.split()) >= 2:
            out.append(p)
    return out

df["author_list"] = df["authors"].apply(parse_authors)
author_to_idx = {}
for i, auths in enumerate(df["author_list"].values):
    for a in auths:
        author_to_idx.setdefault(a, []).append(i)

author_rows = []
for a, idxs in tqdm(author_to_idx.items(), desc="authors"):
    if len(idxs) < 4 or len(idxs) > 40:
        continue
    sub = df.iloc[idxs]
    pre = sub[sub["year"] <= 2022]
    post = sub[sub["year"] >= 2023]
    if len(pre) < 2 or len(post) < 2:
        continue
    def mean_sim(ii):
        V = E[ii]
        S = V @ V.T
        n = len(ii)
        return float((S.sum() - np.trace(S)) / (n*(n-1)))
    d_sem = mean_sim(post.index.to_numpy()) - mean_sim(pre.index.to_numpy())
    d_mattr = post["mattr"].mean() - pre["mattr"].mean()
    d_ttr = post["ttr"].mean() - pre["ttr"].mean()
    fields = sub["field"].value_counts()
    author_rows.append((a, len(sub), len(pre), len(post), fields.index[0], d_sem, d_mattr, d_ttr))
auth = pd.DataFrame(author_rows, columns=["author","n_papers","n_pre","n_post","field","d_sem","d_mattr","d_ttr"])
auth.to_csv(os.path.join(RESULTS, "author_panel.csv"), index=False)
print("authors in panel:", len(auth))
print(auth.groupby("field")[["d_sem","d_mattr","d_ttr"]].agg(["mean","count"]).to_string())

df["emb_idx"] = np.arange(len(df))
df.drop(columns=["author_list"]).to_parquet(os.path.join(RESULTS, "sample_metrics.parquet"), index=False)
print("\nDONE")
