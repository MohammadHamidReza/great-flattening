import duckdb, json, os, sys

PARQUET = os.environ.get("ARXIV_PARQUET", "data/arxiv_metadata.parquet")
con = duckdb.connect()
print("file exists:", os.path.exists(PARQUET), "size:", os.path.getsize(PARQUET) if os.path.exists(PARQUET) else None)
print("\nSCHEMA")
print(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{PARQUET}')").df().to_string())
print("\nCOUNT")
print(con.execute(f"SELECT count(*) AS n FROM read_parquet('{PARQUET}')").fetchone())
print("\nSAMPLE ROW")
row = con.execute(f"SELECT * FROM read_parquet('{PARQUET}') LIMIT 1").fetchdf().to_dict(orient="records")
print(json.dumps(row, indent=2, default=str)[:4000])
print("\nTOP PRIMARY CATEGORIES")
print(con.execute(f"""
    SELECT primary_category, count(*) AS n
    FROM read_parquet('{PARQUET}')
    GROUP BY 1 ORDER BY 2 DESC LIMIT 40
""").df().to_string())
print("\nDATE RANGE / NULLS")
cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{PARQUET}')").fetchall()]
date_col = next((c for c in ["first_version_date","submitted_date","update_date","created_date","publication_date"] if c in cols), None)
print("date column:", date_col)
if date_col:
    print(con.execute(f"SELECT min({date_col}), max({date_col}), count(*) FILTER (WHERE {date_col} IS NULL) FROM read_parquet('{PARQUET}')").fetchone())
if "abstract" in cols:
    print(con.execute(f"""
        SELECT count(*) AS n,
               count(*) FILTER (WHERE abstract IS NULL OR length(abstract)=0) AS null_abs,
               quantile_cont(length(abstract), [0.01,0.1,0.5,0.9,0.99]) AS q
        FROM read_parquet('{PARQUET}')
    """).fetchone())
