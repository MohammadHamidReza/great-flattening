import duckdb, os, json, sys

PARQUET = os.environ.get("ARXIV_PARQUET", "data/arxiv_metadata.parquet")
OUT = os.environ.get("RESULTS_DIR", "results")
os.makedirs(OUT, exist_ok=True)

con = duckdb.connect()
con.execute("PRAGMA threads=8")

FIELD_CASE = """
CASE
  WHEN primary_category = 'cs.CL' THEN 'NLP'
  WHEN primary_category = 'cs.AI' THEN 'AI'
  WHEN primary_category = 'cs.LG' THEN 'ML'
  WHEN primary_category = 'cs.CV' THEN 'CV'
  WHEN primary_category = 'cs.RO' THEN 'ROBOTICS'
  WHEN primary_category LIKE 'math.%' THEN 'MATH'
  WHEN primary_category IN ('hep-th','hep-ph','quant-ph','gr-qc','astro-ph.CO','astro-ph.GA','cond-mat.str-el','cond-mat.mes-hall') THEN 'PHYSICS'
  WHEN primary_category IN ('cs.SE','cs.CR','cs.DC','cs.PL') THEN 'SYSTEMS'
  ELSE 'OTHER'
END
"""

BASE = f"""
WITH raw AS (
  SELECT paper_id, title, authors, abstract, primary_category, categories,
         first_version_date, comments,
         year(first_version_date) AS year,
         {FIELD_CASE} AS field
  FROM read_parquet('{PARQUET}')
  WHERE abstract IS NOT NULL
    AND length(abstract) BETWEEN 200 AND 4000
    AND first_version_date >= DATE '2010-01-01'
    AND first_version_date < DATE '2026-09-01'
    AND (comments IS NULL OR lower(comments) NOT LIKE '%withdraw%')
    AND (primary_category = 'cs.CL' OR primary_category = 'cs.AI' OR primary_category = 'cs.LG'
         OR primary_category = 'cs.CV' OR primary_category = 'cs.RO'
         OR primary_category LIKE 'math.%'
         OR primary_category IN ('hep-th','hep-ph','quant-ph','gr-qc','astro-ph.CO','astro-ph.GA','cond-mat.str-el','cond-mat.mes-hall')
         OR primary_category IN ('cs.SE','cs.CR','cs.DC','cs.PL'))
),
dedup AS (
  SELECT *, row_number() OVER (PARTITION BY hash(abstract) ORDER BY first_version_date) AS dup
  FROM raw
),
clean AS (
  SELECT * FROM dedup WHERE dup = 1
)
"""

print("Materializing filtered corpus...")
con.execute("CREATE OR REPLACE TEMP TABLE corpus AS " + BASE + " SELECT * FROM clean")
print("Counting filtered corpus...")
counts = con.execute("SELECT field, year, count(*) AS n FROM corpus GROUP BY 1,2 ORDER BY 1,2").fetchdf()
counts.to_csv(os.path.join(OUT, "counts.csv"), index=False)
print(counts.pivot(index="year", columns="field", values="n").fillna(0).astype(int).to_string())

print("\nBuilding stratified sample (1,200 per field-year)...")
sample_sql = f"""
SELECT paper_id, title, authors, abstract, primary_category, categories,
       first_version_date, year, field
FROM (
  SELECT *, row_number() OVER (PARTITION BY field, year ORDER BY hash(paper_id)) AS rn
  FROM corpus
) WHERE rn <= 1200
"""
sample = con.execute(sample_sql).fetchdf()
sample.to_parquet(os.path.join(OUT, "sample.parquet"), index=False)
print("sample rows:", len(sample))
print(sample.groupby(["field","year"]).size().unstack(fill_value=0).to_string())

# Phrase counts over the full filtered corpus (not just sample)
phrases = {
    "delve": "delve", "pivotal": "pivotal", "intricate": "intricate", "realm": "realm",
    "testament": "testament", "underscore": "underscore", "tapestry": "tapestry",
    "seamless": "seamless", "holistic": "holistic", "meticulous": "meticulous",
    "multifaceted": "multifaceted", "nuanced": "nuanced", "notably": "notably",
    "furthermore": "furthermore", "moreover": "moreover", "crucial": "crucial",
    "landscape": "landscape", "navigate": "navigate", "leverage": "leverage",
    "robust": "robust", "comprehensive": "comprehensive", "paradigm": "paradigm",
    "foster": "foster", "showcase": "showcase", "shed_light": "shed light",
    "crucial_role": "plays a crucial role", "important_note": "it is important to note",
    "realm_of": "in the realm of", "testament_to": "a testament to",
    "forefront": "at the forefront", "cutting_edge": "cutting-edge",
    "game_changer": "game-changer", "revolutionize": "revolutionize",
    "unprecedented": "unprecedented", "transformative": "transformative",
    "pivotal_role": "pivotal role", "intricate_interplay": "intricate interplay",
    "rich_tapestry": "rich tapestry", "delve_into": "delve into",
    "underscore_importance": "underscore the importance",
    "navigate_complexities": "navigate the complexities",
    "seamless_integration": "seamless integration", "holistic_approach": "holistic approach",
}
select_parts = ["field", "year", "count(*) AS n"]
for alias, phrase in phrases.items():
    esc = phrase.replace("'", "''")
    select_parts.append(f"sum(CASE WHEN lower(abstract) LIKE '%{esc}%' THEN 1 ELSE 0 END) AS {alias}")
phrase_sql = "SELECT " + ", ".join(select_parts) + " FROM corpus GROUP BY 1,2 ORDER BY 1,2"
print("\nComputing phrase counts...")
ph = con.execute(phrase_sql).fetchdf()
ph.to_csv(os.path.join(OUT, "phrase_counts.csv"), index=False)
print(ph.head().to_string())
print("\nDONE")

