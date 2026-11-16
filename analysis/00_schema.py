import duckdb, sys, json, os

path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ARXIV_PARQUET", "data/arxiv_metadata.parquet")
con = duckdb.connect()
print(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").df().to_string())
print("\nCOUNT", con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone())
row = con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 1").fetchdf().to_dict(orient="records")[0]
print("\nSAMPLE")
for k, v in row.items():
    s = str(v)
    print(f"{k}: {s[:500]}")
