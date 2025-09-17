"""HighD dataset loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def load_tracks(path: str | Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load one or multiple HighD tracks CSV files into a single dataframe."""

    path = Path(path)
    files: List[Path]
    if path.is_dir():
        files = sorted(p for p in path.glob("*_tracks.csv"))
        if not files:
            raise FileNotFoundError("No *_tracks.csv files found in directory")
    else:
        if not path.exists():
            raise FileNotFoundError(path)
        files = [path]

    dataframes = []
    for file in files:
        df = pd.read_csv(file, usecols=usecols)
        df["source_file"] = file.name
        stem = file.stem.split("_")[0]
        df["recording_id"] = stem
        dataframes.append(df)
    if not dataframes:
        return pd.DataFrame()
    return pd.concat(dataframes, ignore_index=True)
