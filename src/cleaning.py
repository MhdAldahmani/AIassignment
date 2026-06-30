import pandas as pd

SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}


def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin-1")


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(df)
    df = df.drop_duplicates()
    n1 = len(df)
    df = df.drop_duplicates(subset="Review_ID", keep="first")
    n2 = len(df)
    stats = {
        "rows_before": n0,
        "exact_duplicates_dropped": n0 - n1,
        "duplicate_review_id_dropped": n1 - n2,
        "rows_after": n2,
    }
    return df, stats


def parse_year_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_missing = df["Year_Month"] == "missing"
    parts = df["Year_Month"].where(~is_missing).str.split("-", n=1, expand=True)
    df["year"] = pd.to_numeric(parts[0], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")
    df["season"] = df["month"].map(SEASON_BY_MONTH)
    return df


def clean(path) -> tuple[pd.DataFrame, dict]:
    df = load_raw(path)
    df, dedupe_stats = dedupe(df)
    df = parse_year_month(df)
    stats = {
        **dedupe_stats,
        "missing_year_month": int((df["year"].isna()).sum()),
        "missing_year_month_pct": round(float((df["year"].isna()).mean() * 100), 2),
    }
    return df.reset_index(drop=True), stats
