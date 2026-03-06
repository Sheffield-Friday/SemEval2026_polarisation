from pathlib import Path
from typing import List

import pandas as pd
import yaml

from annotate import load_data, parse_args


def combine_st1(df: pd.DataFrame, annotations_dir: Path) -> pd.DataFrame:
    """
    Merges columns from all CSV files in a directory into the main df
    matching on the 'id' column.
    """
    df_copy = df.copy()

    # ensure annotations_dir is Path obj
    annotations_dir = Path(annotations_dir)

    # find all annotation files
    for file_path in annotations_dir.glob("*.csv"):
        print(file_path)
        # load anno csv
        annotation_df = pd.read_csv(file_path)
        annotation_df = annotation_df.drop(columns=["language"])
        # merge into df
        df_copy = df_copy.merge(annotation_df, on="id", how="left")

    return df_copy


if __name__ == "__main__":
    args = parse_args()

    config_file = args.config
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    st1_cfg = config["subtask1"]
    lang_isos = st1_cfg.get("languages")

    data_path = Path(st1_cfg.get("data_path"))
    save_path = Path(st1_cfg.get("save_dir"))
    for lang in lang_isos:
        lang_data_path = save_path / lang
        df = load_data([lang], str(data_path))

        combined_df = combine_st1(df, lang_data_path)

        combined_df.to_csv(f"./combined/{lang}.csv", index=False)
