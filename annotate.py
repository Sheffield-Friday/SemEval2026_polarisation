"""Module for the LLM annotation pipeline."""

import argparse
import time
from pathlib import Path
from string import Formatter
from typing import Dict, List, Optional

import pandas as pd
import yaml

from personas import Person, generate_list_of_people
from prompt_generator import PromptGenerator

# from classifiers.outlines import OutlinesClassifier
from vllm_classifiers.outlines import OutlinesClassifier, OutlinesMultiClassifier

ISO_TO_LANG = {
    "amh": "Amharic",
    "arb": "Arabic",
    "ben": "Bengali",
    "mya": "Burmese",
    "eng": "English",
    "deu": "German",
    "hau": "Hausa",
    "hin": "Hindi",
    "ita": "Italian",
    "khm": "Khmer",
    "nep": "Nepali",
    "ori": "Odia",
    "fas": "Persian",
    "pol": "Polish",
    "pan": "Punjabi",
    "rus": "Russian",
    "spa": "Spanish",
    "swa": "Swahili",
    "tel": "Telugu",
    "tur": "Turkish",
    "urd": "Urdu",
    "zho": "Chinese",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="File path to config yaml file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split you are running: train, dev, or test.",
    )
    parser.add_argument("--st1", action="store_true")
    parser.add_argument("--st2", action="store_true")
    parser.add_argument("--st3", action="store_true")
    return parser.parse_args()


def get_language_filepath(language: str, data_dir: Path):
    """Get the filepath for the csv given the language.

    Args:
        language (str): language to convert to file path.
        data_path (str): string representation of directory all
            language CSVs are stored in.

    Returns:
        Path: path to CSV file for given language.
    """
    return data_dir / (language.strip() + ".csv")


def get_lang_df(lang: str, data_dir: Path):
    data_file = get_language_filepath(lang, data_dir)
    df = pd.read_csv(data_file)
    df["language"] = lang
    return df


def load_data(languages: List[str], data_path: str):
    """Load data into one dataframe with all data points from
        the provided list of langauges.

    Args:
        langauges (List[str]): list of languages to load for annotation.
        data_path (str): string data path of the directory containing all
            individual langauge CSV files.

    Returns:
        pd.DataFrame: dataframe containing all data points to annotate.
    """
    if len(languages) < 1:
        raise ValueError("Must pass at least one language.")
    # get paths for each langauge
    data_dir = Path(data_path)

    dfs = [get_lang_df(lang, data_dir) for lang in languages]

    # concat langauge dfs
    return pd.concat(dfs, ignore_index=True)


def create_prompt(tokenizer, text, sys_prompt, user_prompt) -> List[dict]:
    s_prompt = PromptGenerator.build_system_prompt(sys_prompt)
    u_prompt = PromptGenerator.build_prompt(user_prompt, text)
    return PromptGenerator.create_full_prompt(tokenizer, u_prompt, s_prompt)


def add_person_characteristic(sys_prompt, person):
    if person is None:
        return sys_prompt

    allowed_keys = {"politics", "race", "religion", "orientation_gender"}
    found_keys = [
        field_name
        for _, field_name, _, _ in Formatter().parse(sys_prompt)
        if field_name
    ]
    valid_matches = set(found_keys).intersection(allowed_keys)

    if len(valid_matches) == 0:
        return sys_prompt

    if len(valid_matches) > 1:
        raise ValueError(
            "Prompt should contain no more than one valid characteristic key."
        )

    key_to_replace = valid_matches.pop()
    replacement_val = getattr(person, key_to_replace, None)

    if replacement_val is None:
        raise ValueError(
            f"Key {key_to_replace} should not be None. Please check configs."
        )

    return sys_prompt.replace(f"{{{key_to_replace}}}", str(replacement_val))


def get_person_sys_prompt(sys_prompt, lang, person):
    # add the person thing here
    sp = add_person_characteristic(sys_prompt, person)
    sp = sp.format(LANGUAGE=lang)
    if person:
        sp = f"You are a {str(person)}. {sp}"
    return sp


# TODO: change the config to be parameterised
def annotate(
    df,
    classifier,
    annotator_name,
    sys_prompt,
    user_prompt,
    time_taken=True,
    person=None,
) -> pd.DataFrame:
    print(f"Annotating {len(df)} data points.")
    start_time = time.time()

    # loop through df and classify each data point
    # TODO: might want to change for conf annotation
    preds = [
        str(
            classifier.classify(
                create_prompt(
                    classifier.tokenizer,
                    row["text"],
                    sys_prompt=get_person_sys_prompt(
                        sys_prompt, ISO_TO_LANG[row["language"]], person
                    ),
                    user_prompt=user_prompt,
                )
            ).get("predicted_label")
        ).lower()
        for _, row in df.iterrows()
    ]

    # output time taken
    time_taken = time.time() - start_time
    if time_taken:
        print(f"Annotation took {time_taken // 60}m {time_taken % 60}s")
        print(f"Average time per annotation: {time_taken / len(preds)}s")

    # create anno df w/ list of ids
    anno_df = pd.DataFrame()
    anno_df["id"] = df["id"]
    anno_df["language"] = df["language"]
    anno_df[annotator_name] = preds

    return anno_df


def multi_annotate(
    df,
    classifier,
    annotator_name,
    sys_prompts,
    user_prompts,
    time_taken=True,
    person=None,
):
    print(f"Annotating {len(df)} data points.")

    start_time = time.time()
    if len(sys_prompts) != len(user_prompts) or len(sys_prompts) != len(
        classifier.class_labels
    ):
        raise ValueError(
            "System prompts, user prompts, and class labels length should all be the same."
        )

    # create message_lists
    # one list of messages per data point (row)
    all_messages_list = [
        [
            create_prompt(
                classifier.tokenizer,
                row["text"],
                sys_prompt=get_person_sys_prompt(
                    sys_prompt, ISO_TO_LANG[row["language"]], person
                ),
                user_prompt=user_prompt,
            )
            for sys_prompt, user_prompt in zip(sys_prompts, user_prompts)
        ]
        for _, row in df.iterrows()
    ]

    # make predictions
    preds = [
        classifier.classify(messages_list).get("predicted_label")
        for messages_list in all_messages_list
    ]

    # output time taken
    time_taken = time.time() - start_time
    if time_taken:
        print(f"Annotation took {time_taken // 60}m {time_taken % 60}s")
        print(f"Average time per annotation: {time_taken / len(preds)}s")

    # create anno df w/ list of ids
    anno_df = pd.DataFrame()
    anno_df["id"] = df["id"]
    anno_df["language"] = df["language"]
    anno_df[annotator_name] = preds

    return anno_df


def save_annotations(df: pd.DataFrame, save_dir: Path, anno_name: str):
    for lang, group in df.groupby("language"):
        # create lang dir if needed
        lang_path = save_dir / str(lang)
        lang_path.mkdir(parents=True, exist_ok=True)

        # save
        save_path = lang_path / (anno_name + ".csv")
        group.to_csv(save_path, index=False)


def save_dict_to_txt(data_dict, filename):
    """Saves a list of dictionaries to a text file in 'key = value' format."""
    with open(filename, "w", encoding="utf-8") as f:
        for key, value in data_dict.items():
            f.write(f"{key} = {value}\n")


def st1(config, split):
    anno_name = config["annotator_name"]

    # load data to annotate
    st1_cfg = config["subtask1"]
    df = load_data(
        st1_cfg.get("languages"), st1_cfg.get("data_path").format(split=split)
    )
    # df.to_csv("test.csv", index=False)

    # load option for prompt persona
    if config["use_personas"]:
        # people = [Person() for _ in range(config["num_personas"])]
        people, _ = generate_list_of_people(config["num_personas"])
        annotators = {
            anno_name + f"-annotator_{i}": person for i, person in enumerate(people)
        }
    else:
        annotators = {anno_name: None}

    # keep track of annotator identities
    anno_save_path = Path(f"./annotators/st1/{split}")
    anno_save_path.mkdir(parents=True, exist_ok=True)
    save_dict_to_txt(annotators, str(anno_save_path / f"{anno_name}.txt"))

    # annotate data
    classifier = OutlinesClassifier(
        config["model_name"],
        st1_cfg["class_labels"],
        st1_cfg["conf_labels"],
        num_gpus=config["num_gpus"],
        max_model_len=config["max_model_len"],
    )

    # set save dir path
    save_dir = Path(st1_cfg["save_dir"].format(split=split))

    for anno_name, person in annotators.items():
        anno_df = annotate(
            df,
            classifier,
            annotator_name=anno_name,
            sys_prompt=st1_cfg["sys_prompt"],
            user_prompt=st1_cfg["user_prompt"],
            person=person,
        )

        # save the annotations
        save_annotations(anno_df, save_dir, anno_name)


def st2(config, split):
    anno_name = config["annotator_name"]

    # load data to annotate
    st2_cfg = config["subtask2"]
    df = load_data(
        st2_cfg.get("languages"), st2_cfg.get("data_path").format(split=split)
    )

    # load option for prompt persona
    if config["use_personas"]:
        # people = [Person() for _ in range(config["num_personas"])]
        people, _ = generate_list_of_people(config["num_personas"])
        annotators = {
            anno_name + f"-annotator_{i}": person for i, person in enumerate(people)
        }
    else:
        annotators = {anno_name: None}

    # keep track of annotator identities
    anno_save_path = Path(f"./annotators/st2/{split}")
    anno_save_path.mkdir(parents=True, exist_ok=True)
    save_dict_to_txt(annotators, str(anno_save_path / f"{anno_name}.txt"))

    label_prefixes = ["pol", "race", "rel", "gen", "other"]

    # annotate data
    classifier = OutlinesMultiClassifier(
        config["model_name"],
        [st2_cfg["class_labels"] for _ in range(len(label_prefixes))],
        st2_cfg["conf_labels"],
        num_gpus=config["num_gpus"],
        max_model_len=config["max_model_len"],
    )

    # set save dir path
    save_dir = Path(st2_cfg["save_dir"].format(split=split))

    sys_prompts = [
        st2_cfg[f"{label_prefix}_sys_prompt"] for label_prefix in label_prefixes
    ]
    user_prompts = [
        st2_cfg[f"{label_prefix}_user_prompt"] for label_prefix in label_prefixes
    ]
    for anno_name, person in annotators.items():
        anno_df = multi_annotate(
            df,
            classifier,
            annotator_name=anno_name,
            sys_prompts=sys_prompts,
            user_prompts=user_prompts,
            person=person,
        )

        # save the annotations
        save_annotations(anno_df, save_dir, anno_name)

def st3(config, split):
    anno_name = config["annotator_name"]

    # load data to annotate
    st3_cfg = config["subtask3"]
    df = load_data(
        st3_cfg.get("languages"), st3_cfg.get("data_path").format(split=split)
    )

    # load option for prompt persona
    if config["use_personas"]:
        # people = [Person() for _ in range(config["num_personas"])]
        people, _ = generate_list_of_people(config["num_personas"])
        annotators = {
            anno_name + f"-annotator_{i}": person for i, person in enumerate(people)
        }
    else:
        annotators = {anno_name: None}

    # keep track of annotator identities
    anno_save_path = Path(f"./annotators/st3/{split}")
    anno_save_path.mkdir(parents=True, exist_ok=True)
    save_dict_to_txt(annotators, str(anno_save_path / f"{anno_name}.txt"))

    label_prefixes = ["stereo", "vil", "dehum", "ela", "leu", "inv"]

    # annotate data
    classifier = OutlinesMultiClassifier(
        config["model_name"],
        [st3_cfg["class_labels"] for _ in range(len(label_prefixes))],
        st3_cfg["conf_labels"],
        num_gpus=config["num_gpus"],
        max_model_len=config["max_model_len"],
    )

    # set save dir path
    save_dir = Path(st3_cfg["save_dir"].format(split=split))

    sys_prompts = [
        st3_cfg[f"{label_prefix}_sys_prompt"] for label_prefix in label_prefixes
    ]
    user_prompts = [
        st3_cfg[f"{label_prefix}_user_prompt"] for label_prefix in label_prefixes
    ]
    for anno_name, person in annotators.items():
        anno_df = multi_annotate(
            df,
            classifier,
            annotator_name=anno_name,
            sys_prompts=sys_prompts,
            user_prompts=user_prompts,
            person=person,
        )

        # save the annotations
        save_annotations(anno_df, save_dir, anno_name)


def main():
    args = parse_args()

    if args.split is None or args.split not in ["train", "dev", "test"]:
        raise ValueError(
            "You must enter a value for --split and it must be one of [train, dev, test]."
        )

    # load config
    config_file = args.config
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    if not args.st1 and not args.st2 and not args.st3:
        raise ValueError("Pass one of the following arguments: --st1, --st2, --st3")

    if args.st1:
        print("Running SubTask1")
        st1(config, args.split)

    if args.st2:
        print("Running SubTask2")
        st2(config, args.split)

    if args.st3:
        print("Running SubTask3")
        st3(config, args.split)


if __name__ == "__main__":
    main()
