import argparse
import ast
import traceback
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
from effiara.agreement import inter_annotator_agreement_krippendorff
from effiara.label_generators import DefaultLabelGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="File path to config yaml file.",
    )
    return parser.parse_args()


def load_data(data_dir: Path, languages: List[str]):
    dfs = [pd.read_csv(data_dir / (lang + ".csv")) for lang in languages]
    return pd.concat(dfs)


def eval_objects(x):
    try:
        # only parse if it looks like a list or dict
        if isinstance(x, str) and (x.startswith("[") or x.startswith("{")):  # } ]
            return ast.literal_eval(x)
        return x
    except (ValueError, SyntaxError):
        # handle cases where the string is malformed / normal string
        return x


def combine_st1(
    df: pd.DataFrame, annotations_dir: Path, annotators: List[str], languages: List[str]
):
    df_copy = df.copy()
    annotations_dir = Path(annotations_dir)

    # add all annotators to the df_copy
    for annotator in annotators:
        anno_dfs = [
            pd.read_csv(annotations_dir / lang / (annotator + ".csv"))
            for lang in languages
        ]
        annotation_df = pd.concat(anno_dfs)
        annotation_df = annotation_df.drop(columns=["language"])
        df_copy = df_copy.merge(annotation_df, on="id", how="left")

    return df_copy


LLAMA_ANNOTATORS = ["llama-3-70b-base"] + [
    f"llama-3-70b-pol-personas-annotator_{i}" for i in range(10)
]
#  LLAMA_ANNOTATORS = [
#  f"llama-3-70b-pol-personas-annotator_{i}" for i in range(10)
#  ]
QWEN_ANNOTATORS = ["qwen25-72b-base"] + [
    f"qwen25-72b-pol-personas-annotator_{i}" for i in range(10)
]
#  QWEN_ANNOTATORS = [
#  f"qwen25-72b-pol-personas-annotator_{i}" for i in range(10)
#  ]
GEMMA_ANNOTATORS = ["gemma-3-27b-it-base"] + [
    f"gemma-3-27b-it-pol-personas-annotator_{i}" for i in range(10)
]
#  GEMMA_ANNOTATORS = [
#  f"gemma-3-27b-it-pol-personas-annotator_{i}" for i in range(10)
#  ]

BASE_ANNOTATORS = ["llama-3-70b-base", "qwen25-72b-base", "gemma-3-27b-it-base"]


def iaa_with_fallback(df, annotators, label_mapping):
    try:
        return inter_annotator_agreement_krippendorff(df, annotators, label_mapping)
    except ValueError as e:
        print("error encountered...")
        traceback.print_exc()
        return 1.0


def get_agreement_st1(df, languages, annotators):
    label_mapping = {"no": 0, "yes": 1}

    ldfs = [df[df["id"].str.startswith(lang)] for lang in languages[:-1]]
    return [iaa_with_fallback(ldf, annotators, label_mapping) for ldf in ldfs] + [
        iaa_with_fallback(df, annotators, label_mapping)
    ]


def get_agreement_multi(df, languages, annotators, labels):
    label_mapping = {"no": 0, "yes": 1}

    ldfs = [df[df["id"].str.startswith(lang)] for lang in languages[:-1]]
    results = []
    for label in labels:
        current_annotators = [f"{annotator}_{label}" for annotator in annotators]
        to_append = [
            iaa_with_fallback(ldf, current_annotators, label_mapping) for ldf in ldfs
        ]
        to_append.append(iaa_with_fallback(df, current_annotators, label_mapping))
        results.append(to_append)

    results_matrix = np.array(results)

    avg_agreement_vector = np.mean(results_matrix, axis=0)
    return avg_agreement_vector.tolist()


if __name__ == "__main__":
    # load args
    args = parse_args()

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    # get anntotators
    annotators = LLAMA_ANNOTATORS + QWEN_ANNOTATORS + GEMMA_ANNOTATORS
    #  annotators = BASE_ANNOTATORS
    #  annotators = LLAMA_ANNOTATORS + QWEN_ANNOTATORS
    #  annotators = ["llama-3-70b-base", "qwen25-72b-base", "gemma-3-27b-it-base"]
    print(annotators)

    # paths for each subtask...
    st1_data_path = config["st1"]["data_path_pol"]
    st3_data_path = "st3_with_all_aggregations.csv"

    # add label names for each subtask

    st1_res_df = pd.DataFrame()
    st1_res_df["languages"] = config["st1"]["languages"] + ["Overall"]

    # run label aggregations
    label_generator = DefaultLabelGenerator(
        annotators, label_mapping={"no": 0, "yes": 1}, label_suffixes=[""]
    )
    qwen_pol = [f"qwen25-72b-pol-personas-annotator_{i}" for i in range(10)]
    llama_pol = [f"llama-3-70b-pol-personas-annotator_{i}" for i in range(10)]
    gemma_pol = [f"gemma-3-27b-it-pol-personas-annotator_{i}" for i in range(10)]

    # subtask 1

    qwen_hate_st1 = [f"qwen25-72b-annotator_{i}" for i in range(10)]
    llama_hate_st1 = [f"llama-3-70b-annotator_{i}" for i in range(10)]
    gemma_hate_st1 = [f"gemma-3-27b-it-annotator_{i}" for i in range(10)]

    st1_df = pd.read_csv(st1_data_path)
    st1_res_df["base"] = get_agreement_st1(
        st1_df, st1_res_df["languages"].tolist(), BASE_ANNOTATORS
    )

    all_pol_annotators = llama_pol + qwen_pol + gemma_pol
    st1_res_df["pol_all"] = get_agreement_st1(
        st1_df, st1_res_df["languages"].tolist(), all_pol_annotators
    )
    st1_res_df["pol_llama"] = get_agreement_st1(
        st1_df, st1_res_df["languages"].tolist(), llama_pol
    )
    st1_res_df["pol_qwen"] = get_agreement_st1(
        st1_df, st1_res_df["languages"].tolist(), qwen_pol
    )

    st1_res_df["pol_gemma"] = get_agreement_st1(
        st1_df, st1_res_df["languages"].tolist(), gemma_pol
    )

    print(st1_res_df.to_markdown(index=False))
    input("subtask 1 pol annotations")

    # hate
    st1_res_df["base"] = get_agreement_st1(
        st1_df, st1_res_df["languages"].tolist(), BASE_ANNOTATORS
    )

    all_hate_annotators_st1 = llama_hate_st1 + qwen_hate_st1 + gemma_hate_st1
    st1_res_df["hate_all"] = get_agreement_st1(
        st1_df, st1_res_df["languages"].tolist(), all_hate_annotators_st1
    )
    st1_res_df["hate_llama"] = get_agreement_st1(
        st1_df, st1_res_df["languages"].tolist(), llama_hate_st1
    )
    st1_res_df["hate_qwen"] = get_agreement_st1(
        st1_df, st1_res_df["languages"].tolist(), qwen_hate_st1
    )

    st1_res_df["hate_gemma"] = get_agreement_st1(
        st1_df, st1_res_df["languages"].tolist(), gemma_hate_st1
    )

    print(st1_res_df.to_markdown(index=False))
    input("subtask 1 hate annotations")

    # subtask 2
    st2_data_path = config["st2"]["data_path_pol"]

    st2_res_df = pd.DataFrame()
    st2_res_df["languages"] = config["st2"]["languages"] + ["Overall"]

    st2_df = pd.read_csv(config["st2"]["data_path_pol"])
    st2_res_df["base"] = get_agreement_multi(
        st2_df,
        st2_res_df["languages"].tolist(),
        BASE_ANNOTATORS,
        config["st2"]["labels"],
    )

    st2_res_df["pol_all"] = get_agreement_multi(
        st2_df,
        st2_res_df["languages"].tolist(),
        all_pol_annotators,
        config["st2"]["labels"],
    )
    st2_res_df["pol_llama"] = get_agreement_multi(
        st2_df, st2_res_df["languages"].tolist(), llama_pol, config["st2"]["labels"]
    )
    st2_res_df["pol_qwen"] = get_agreement_multi(
        st2_df, st2_res_df["languages"].tolist(), qwen_pol, config["st2"]["labels"]
    )

    st2_res_df["pol_gemma"] = get_agreement_multi(
        st2_df, st2_res_df["languages"].tolist(), gemma_pol, config["st2"]["labels"]
    )

    print(st2_res_df.to_markdown(index=False))
    input("subtask 2 pol annotations")

    # hate
    qwen_hate_st2 = [f"qwen25-72b-hate-personas-annotator_{i}" for i in range(10)]
    llama_hate_st2 = [f"llama-3-70b-hate-personas-annotator_{i}" for i in range(10)]
    gemma_hate_st2 = [f"gemma-3-27b-it-hate-personas-annotator_{i}" for i in range(10)]

    st2_df = pd.read_csv(config["st2"]["data_path_pol"])

    all_hate_annotators_st2 = llama_hate_st2 + qwen_hate_st2 + gemma_hate_st2
    st2_res_df["hate_all"] = get_agreement_multi(
        st2_df,
        st2_res_df["languages"].tolist(),
        all_hate_annotators_st2,
        config["st2"]["labels"],
    )
    st2_res_df["hate_llama"] = get_agreement_multi(
        st2_df,
        st2_res_df["languages"].tolist(),
        llama_hate_st2,
        config["st2"]["labels"],
    )
    st2_res_df["hate_qwen"] = get_agreement_multi(
        st2_df, st2_res_df["languages"].tolist(), qwen_hate_st2, config["st2"]["labels"]
    )

    st2_res_df["hate_gemma"] = get_agreement_multi(
        st2_df,
        st2_res_df["languages"].tolist(),
        gemma_hate_st2,
        config["st2"]["labels"],
    )
    print(st2_res_df.to_markdown(index=False))
    input("subtask 2 hate annotations")

    # subtask 3
    st3_data_path = config["st3"]["data_path_pol"]

    st3_res_df = pd.DataFrame()
    st3_res_df["languages"] = config["st3"]["languages"] + ["Overall"]

    st3_df = pd.read_csv(config["st3"]["data_path_pol"])

    st3_res_df["pol_all"] = get_agreement_multi(
        st3_df,
        st3_res_df["languages"].tolist(),
        all_pol_annotators,
        config["st3"]["labels"],
    )
    st3_res_df["pol_llama"] = get_agreement_multi(
        st3_df,
        st3_res_df["languages"].tolist(),
        LLAMA_ANNOTATORS,
        config["st3"]["labels"],
    )
    st3_res_df["pol_qwen"] = get_agreement_multi(
        st2_df,
        st3_res_df["languages"].tolist(),
        QWEN_ANNOTATORS,
        config["st3"]["labels"],
    )

    st3_res_df["pol_gemma"] = get_agreement_multi(
        st3_df,
        st3_res_df["languages"].tolist(),
        GEMMA_ANNOTATORS,
        config["st3"]["labels"],
    )

    print(st3_res_df.to_markdown(index=False))
    input("subtask 2 pol annotations")
