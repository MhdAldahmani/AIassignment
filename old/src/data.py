from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pandas.api.types import is_string_dtype


MISSING_TOKENS = {"", "missing", "nan", "none", "null", "na", "n/a"}
REQUIRED_RAW_COLUMNS = {
    "Review_ID",
    "Rating",
    "Year_Month",
    "Reviewer_Location",
    "Review_Text",
    "Branch",
}


def locate_dataset(search_dir: str | Path | None = None) -> Path:
    search_path = Path(search_dir or Path.cwd())
    candidates = sorted(search_path.glob("*.csv"))
    disney_candidates = [path for path in candidates if "disney" in path.name.lower()]
    if disney_candidates:
        return disney_candidates[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError("Could not uniquely identify the Disneyland reviews CSV.")


def load_reviews_csv(path: str | Path) -> tuple[pd.DataFrame, str]:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            return df, encoding
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise ValueError(f"Unable to load dataset from {path}")


def clean_review_text(text: object) -> str:
    if pd.isna(text):
        return ""
    cleaned = str(text).replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(
        r"\b(\w+?)\s{2,}(t|s|re|ve|ll|d|m)\b",
        r"\1'\2",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def month_to_season(month: object) -> str | pd.NA:
    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }
    if pd.isna(month):
        return pd.NA
    return season_map.get(int(month), pd.NA)


def clean_reviews_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    reviews_df = raw_df.copy()
    reviews_df.columns = reviews_df.columns.str.strip()

    missing_columns = REQUIRED_RAW_COLUMNS.difference(reviews_df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    for column in reviews_df.columns:
        if is_string_dtype(reviews_df[column]) or reviews_df[column].dtype == object:
            reviews_df[column] = reviews_df[column].astype("string").str.strip()

    reviews_df = reviews_df.drop_duplicates().copy()

    reviews_df["_text_length_chars"] = reviews_df["Review_Text"].fillna("").str.len()
    reviews_df["_truncated_variant"] = reviews_df["Review_Text"].fillna("").str.endswith("...More")
    reviews_df = (
        reviews_df.sort_values(
            ["Review_ID", "_truncated_variant", "_text_length_chars"],
            ascending=[True, True, False],
        )
        .drop_duplicates(subset=["Review_ID"], keep="first")
        .copy()
    )
    reviews_df = reviews_df.drop(columns=["_text_length_chars", "_truncated_variant"])

    reviews_df["Branch"] = reviews_df["Branch"].fillna("").astype("string").str.strip()

    reviews_df["Reviewer_Location"] = (
        reviews_df["Reviewer_Location"].fillna("").astype("string").str.strip()
    )
    reviews_df["Reviewer_Location"] = reviews_df["Reviewer_Location"].mask(
        reviews_df["Reviewer_Location"].str.lower().isin(MISSING_TOKENS),
        "Unknown",
    )

    reviews_df["Rating"] = pd.to_numeric(reviews_df["Rating"], errors="coerce").astype("Int64")

    year_month_clean = reviews_df["Year_Month"].fillna("").astype("string").str.strip()
    year_month_clean = year_month_clean.mask(year_month_clean.str.lower().isin(MISSING_TOKENS))
    reviews_df["Review_Date"] = pd.to_datetime(year_month_clean, format="%Y-%m", errors="coerce")
    reviews_df["Year_Month"] = reviews_df["Review_Date"].dt.strftime("%Y-%m").astype("string")

    reviews_df["Review_Text"] = reviews_df["Review_Text"].map(clean_review_text).astype("string")
    reviews_df["Year"] = reviews_df["Review_Date"].dt.year.astype("Int64")
    reviews_df["Month"] = reviews_df["Review_Date"].dt.month.astype("Int64")
    reviews_df["Month_Name"] = reviews_df["Review_Date"].dt.month_name().astype("string")
    reviews_df["Season"] = reviews_df["Month"].map(month_to_season).astype("string")
    reviews_df["Review_Length"] = reviews_df["Review_Text"].str.split().str.len().astype("Int64")
    reviews_df["Rating_Sentiment"] = pd.Series(
        pd.cut(
            reviews_df["Rating"].astype("float64"),
            bins=[0, 2, 3, 5],
            labels=["negative", "neutral", "positive"],
            include_lowest=True,
        ),
        index=reviews_df.index,
    ).astype("string")

    return reviews_df.reset_index(drop=True)


def load_clean_reviews(
    path: str | Path | None = None,
    *,
    search_dir: str | Path | None = None,
) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else locate_dataset(search_dir)
    raw_df, _ = load_reviews_csv(csv_path)
    return clean_reviews_dataframe(raw_df)
