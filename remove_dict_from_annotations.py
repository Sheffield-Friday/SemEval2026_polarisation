import ast
from pathlib import Path

import pandas as pd

ANNOTATION_DIR = "./annotations"


def eval_objects(x):
    try:
        # only parse if it looks like a list or dict
        if isinstance(x, str) and (x.startswith("[") or x.startswith("{")):  # } ]
            return ast.literal_eval(x)
        return x
    except (ValueError, SyntaxError):
        # handle cases where the string is malformed / normal string
        return x


def change_dfs(csv_dfs):
    # check the value in the name column (can derive from filename)
    for csv_path, df in csv_dfs.items():
        anno_name = csv_path.stem
        df[anno_name] = df[anno_name].apply(eval_objects)
        sample_val = df[anno_name].iloc[0]
        print(f"Before: {type(sample_val)}")

        if isinstance(sample_val, dict):
            df[anno_name] = df[anno_name].apply(lambda x: x["predicted_label"])

        sample_val = df[anno_name].iloc[0]
        print(f"After: {type(sample_val)}")

        df.to_csv(csv_path, index=False)


def check_dfs(csv_dfs):
    for csv_path, df in csv_dfs.items():
        anno_name = csv_path.stem
        df[anno_name] = df[anno_name].apply(eval_objects)
        sample_val = df[anno_name].iloc[0]
        if not isinstance(sample_val, list) and not isinstance(sample_val, str):
            print(f"{csv_path} is not the correct data type, it is {type(sample_val)}.")


if __name__ == "__main__":
    # get a dict of file paths and dataframes
    print("Loading data...")
    root_dir = Path(ANNOTATION_DIR)
    csv_paths = list(root_dir.rglob("*.csv"))

    csv_dfs = {csv_path: pd.read_csv(csv_path) for csv_path in csv_paths}

    print("Data loaded.")

    print("Processing and changing data...")
    change_dfs(csv_dfs)
    print("Data processed and changed.")

    print("Reloading data...")
    root_dir = Path(ANNOTATION_DIR)
    csv_paths = list(root_dir.rglob("*.csv"))

    csv_dfs = {csv_path: pd.read_csv(csv_path) for csv_path in csv_paths}

    print("Data loaded.")

    print("Checking data...")
    check_dfs(csv_dfs)
    print("Data checked.")
