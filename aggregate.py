import argparse
import ast
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
from crowdkit.aggregation import KOS, MACE, MMSR, DawidSkene, MajorityVote
from effiara.annotator_reliability import Annotations
from effiara.label_generators import DefaultLabelGenerator
from sklearn.metrics import f1_score

random.seed(42)

# list of annotator names
LLAMA_ANNOTATORS = ["llama-3-70b-base"] + [
    f"llama-3-70b-pol-personas-annotator_{i}" for i in range(10)
] #+ [f"llama-3-70b-hate-personas-annotator_{i}" for i in range(10)]
#  LLAMA_ANNOTATORS = [
    #  f"llama-3-70b-pol-personas-annotator_{i}" for i in range(10)
#  ]
QWEN_ANNOTATORS = ["qwen25-72b-base"] + [
    f"qwen25-72b-pol-personas-annotator_{i}" for i in range(10)
] #+ [f"qwen25-72b-hate-personas-annotator_{i}" for i in range(10)]
#  QWEN_ANNOTATORS = [
    #  f"qwen25-72b-pol-personas-annotator_{i}" for i in range(10)
#  ]
GEMMA_ANNOTATORS = ["gemma-3-27b-it-base"] + [
    f"gemma-3-27b-it-pol-personas-annotator_{i}" for i in range(10)
] #+ [f"gemma-3-27b-it-hate-personas-annotator_{i}" for i in range(10)]
#  GEMMA_ANNOTATORS = [
    #  f"gemma-3-27b-it-pol-personas-annotator_{i}" for i in range(10)
#  ]

BASE_ANNOTATORS = ["llama-3-70b-base", "qwen25-72b-base", "gemma-3-27b-it-base"]


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


# combine the dataframes
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


# TODO: add a method to split up annotations
def split_multi_annotations(df, annotators, label_names):
    new_df = df.copy()
    new_df.to_csv("checking_for_nans.csv", index=False)
    # change the annotators columns to literal objects
    for annotator in annotators:
        new_df[annotator] = new_df[annotator].apply(eval_objects)

    # check that label_names len is same as df lists
    # sanity check (not conclusive -- would need to check all)
    print(type(new_df[annotators[0]].iloc[0]))
    print(new_df[annotators[0]].iloc[0])
    print(label_names)
    if not isinstance(new_df[annotators[0]].iloc[0], list):
        raise ValueError("df must contain lists in the annotator columns.")

    if len(new_df[annotators[0]].iloc[0]) != len(label_names):
        raise ValueError("Lists in dataframe do not match the length of label_names.")
    # go through each annotator / label name pairing
    for annotator in annotators:
        new_col_names = [f"{annotator}_{label_name}" for label_name in label_names]
        expanded_data = pd.DataFrame(
            new_df[annotator].tolist(), columns=new_col_names, index=new_df.index
        )

        new_df = pd.concat([new_df, expanded_data], axis=1)

    return new_df


def effi_to_crowdkit(df, task_col: str, workers: List[str], label_mapping):
    ck_df = pd.melt(
        df,
        id_vars=[task_col],
        value_vars=workers,
        var_name="worker",
        value_name="label",
    )

    ck_df = ck_df.rename(columns={task_col: "task"})

    # remap label values
    ck_df["label"] = ck_df["label"].map(label_mapping)

    # TODO: add check for where label has not been provided
    ck_df = ck_df.dropna(subset=["label"])

    return ck_df


# run label aggregations
def agg_separate_langs(df, langs, aggregator) -> pd.Series:
    # split into languages
    ldfs = [df[df["task"].str.startswith(lang)] for lang in langs]
    # get the preds
    pred_list: List[pd.Series] = [aggregator().fit_predict(ldf) for ldf in ldfs]

    # merge preds
    return pd.concat(pred_list)


# compare answer(s) with gold standard
if __name__ == "__main__":
    # load args
    args = parse_args()

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    # get anntotators
    annotators = LLAMA_ANNOTATORS + QWEN_ANNOTATORS + GEMMA_ANNOTATORS
    #  annotators = BASE_ANNOTATORS
    #  annotators = LLAMA_ANNOTATORS + QWEN_ANNOTATORS
    random.shuffle(annotators)
    #  annotators = ["llama-3-70b-base", "qwen25-72b-base", "gemma-3-27b-it-base"]
    print(annotators)

    # load data
    # check if need to combine
    if config["combine"]:
        # load the df
        data_dir = Path(config["data_dir"])
        df = load_data(data_dir, config["languages"])

        # combine
        df = combine_st1(df, config["annotations_dir"], annotators, config["languages"])
        # save
        df.to_csv(config["combined_data_path"], index=False)
    else:
        df = pd.read_csv(config["combined_data_path"])

    # run label aggregations
    label_generator = DefaultLabelGenerator(
        annotators, label_mapping={"no": 0, "yes": 1}, label_suffixes=[""]
    )
    #  annotations = Annotations(df, agreement_suffix="", label_generator=label_generator)
    #  annotations.display_agreement_heatmap(display_upper=True)

    # TODO: add in annotation separation for st2/3 here
    if config["subtask"] != "st1":
        df = split_multi_annotations(df, annotators, config["multi_label_names"])

    # run the aggregations
    agg_methods = [
        {"class": DawidSkene, "col_name": "dawid-skene"},
        {"class": MajorityVote, "col_name": "majority-vote"},
        #  {"class": MACE, "col_name": "mace"},
        #  {"class": KOS, "col_name": "kos"},
        #  {"class": MMSR, "col_name": "mmsr"},
    ]

    
    # TODO: change this so is generic and allows for multi-label
    if config["subtask"] == "st1":
        label_names = [""]
    else:
        label_names = config["multi_label_names"]

    # create results dataframe to hold f1 score for each aggregation method and language
    res_df = pd.DataFrame()
    res_df["languages"] = config["languages"]

    for label in label_names:
        # set vars based on subtask
        if config["subtask"] == "st1":
            suffixed_annotators = annotators
            col_addition = ""
            gold_col = "polarization"
        else:
            suffixed_annotators = [f"{annotator}_{label}" for annotator in annotators]
            col_addition = f"-{label}"
            gold_col = label

        # convert data to crowdkit-friendly format
        ck_df = effi_to_crowdkit(df, "id", suffixed_annotators, label_generator.label_mapping)

        # run aggregation methods for each individual label (or just polarisation if st1)
        for agg in agg_methods:
            print(f"Running {agg['col_name']}.")
            preds = agg["class"]().fit_predict(ck_df)
            # add column with that aggregations to original dataframe
            # use all annotations in aggregation (combine languages)
            df = df.merge(
                preds.rename(agg["col_name"] + col_addition + "-all"),
                left_on="id",
                right_index=True,
                how="left",
            )
            # aggregate only separate languages
            preds = agg_separate_langs(ck_df, config["languages"], agg["class"])
            df = df.merge(
                preds.rename(agg["col_name"] + col_addition + "-lang"),
                left_on="id",
                right_index=True,
                how="left",
            )


    # compare with gold standard
    # TODO: make this an st1 function
        # TODO: add a function to do the analysis of st2/3
        if config["test"]:
            print("Running tests...")
            if config["subtask"] != "st1":
                print("Comparing methods with gold-standard")
                dfs = [df[df["id"].str.startswith(lang)] for lang in config["languages"]]
                gold_standard = list(df[gold_col])
                for agg in agg_methods:
                    for suffix in [f"{col_addition}-all", f"{col_addition}-lang"]:
                        col_name = agg["col_name"] + suffix
                        f1s = []
                        for ldf in dfs:
                            y_true = list(ldf[gold_col])
                            y_pred = list(ldf[col_name])
                            f1s.append(f1_score(y_true, y_pred, average="macro"))

                        res_df[col_name] = f1s

    # add overall f1-macro results
    # stack the gold-columns into one y_true
    # TODO: fix this -- need to get the column...
            else:
                df["gold_standard"] = df[gold_col]

                dfs = [df[df["id"].str.startswith(lang)] for lang in config["languages"]]

                # loop through aggregation methods
                for agg in agg_methods:
                    method_name = agg["col_name"]
                    for suffix in ["-all", "-lang"]:
                        f1s = []
                        for ldf in dfs:
                            y_true = np.array(ldf["gold_standard"].tolist())
                            y_pred = np.column_stack([ldf[f"{method_name}{col_addition}{suffix}"].tolist() for label in label_names])
                            f1s.append(f1_score(y_true, y_pred, average="macro"))
                        res_df[f"{method_name}{suffix}"] = f1s
            # TODO: fix this, need to get the column values...
            # TODO: split into languages...
            

    # get both lang and -all
    # stack the columns into one preds


    
    print("saving dfs...")
    df.to_csv(f"x{config['subtask']}_with_all_aggregations.csv", index=False)
    res_df.to_csv(f"xo_results_df_dev_all_{config['subtask']}.csv", index=False)
    print("Done.")

    # output the aggregation stuff
    # loop through aggregation methods
    for agg_method in agg_methods:
        method = agg_method["col_name"]
        for lang_suffix in ["lang", "all"]:
            score = 0
            for label in label_names:
                if label != "":
                    score += res_df[f"{method}-{label}-{lang_suffix}"].mean()
                else:
                    score += res_df[f"{method}-{lang_suffix}"].mean()
            score /= len(label_names)
            print(f"method: {method}, lang: {lang_suffix}, avg: {score}")
