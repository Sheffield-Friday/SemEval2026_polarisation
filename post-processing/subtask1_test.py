"""
Calculates the bp_0.1 aggregation for subtask 1, converts to correct output format and stores in outputs/subtask1/
"""

from pathlib import Path
import pandas as pd

from utils import *

raw_results_folder = "annotations/st1/test/"
outputs_folder = "outputs/subtask_1"
aggregated_results_folder = "outputs/subtask1_aggregated_annotations"
models = ["gemma-3-27b-it-", "llama-3-70b-", "qwen25-72b-"]
annotators = ["base", "hate-personas-annotator_0", "hate-personas-annotator_1", "hate-personas-annotator_2", "hate-personas-annotator_3", "hate-personas-annotator_4",
              "hate-personas-annotator_5", "hate-personas-annotator_6", "hate-personas-annotator_7", "hate-personas-annotator_8", "hate-personas-annotator_9"]

def aggregate(save=False):
    """
    Returns dictionary of {lang_code : aggregated_df} 

    Each aggregated df has the following columns:
    - id
    - base_yes: number of base models who rated the text as polarised
    - base_no: number of base models who rated the text as not polarised
    - personas_yes: number of personas who rated the text as hateful
    - personas_no: number of personas who rated the text as not hateful
    """

    aggregated_dfs = {}

    if save:
        Path(aggregated_results_folder).mkdir(parents=True, exist_ok=True)
    
    for lang_code in lang_codes:
        combined_df = -1
        lang_folder = Path(raw_results_folder + lang_code + "/")
        for filepath in lang_folder.glob("*.csv"):
            if type(combined_df) == int:
                combined_df = pd.read_csv(filepath).drop(columns=["language"])
            else:
                temp = pd.read_csv(filepath).drop(columns=["language"])
                combined_df = combined_df.merge(temp, on="id", how="left")
        
        # validate
        is_valid = validate_df(combined_df, models=models, annotators=annotators)
        if is_valid:
            print(f"{lang_code} results validated")
        else:
            print(f"{lang_code} invalid!")
            exit()

        # aggregate the annotators
        output_df_cols = ["id", "yes_base", "no_base", "yes_personas", "no_personas"]

        # all models with the base annotator ONLY
        combined_df = combined_df.apply(lambda x: get_counts(x, model=models, annotator="base", 
                                            yes_output_col="yes_base", no_output_col="no_base"), axis=1)
        
        # all models with persona annotators ONLY
        combined_df = combined_df.apply(lambda x: get_counts(x, model=models, annotator=annotators[1:], 
                                           yes_output_col="yes_personas", no_output_col="no_personas"), axis=1)
        
        aggregated_dfs[lang_code] = combined_df[output_df_cols]

        if save:
            aggregated_dfs[lang_code].to_csv(aggregated_results_folder + f"/{lang_code}.csv")

    return aggregated_dfs


def get_results(dfs=None):
    """
    Docstring for get_results
    
    :param dfs: Aggregated dataframes as calculated in aggregate()

    Returns dictionary of {lang_code : aggregated_df} 

    Each aggregated df has the following columns:
    - id
    - base_yes: number of base models who rated the text as polarised
    - base_no: number of base modelw who rated the text as not polarised
    - personas_yes: number of personas who rated the text as hateful
    - personas_no: number of personas who rated the text as not hateful

    Plus results columns for each method
    """
    if not dfs:
        dfs = load_aggregated_dfs(aggregated_results_folder + "/LANG_CODE.csv")

    for lang_code in lang_codes:
        dfs[lang_code]["base_maj"] = dfs[lang_code].apply(lambda x: get_majority_rating(x, "base"), axis=1)


        for proportion in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            dfs[lang_code]["personas_" + str(proportion)] = dfs[lang_code].apply(lambda x: get_disagreement_rating(x, "personas", allowed_disagree=proportion), axis=1)

            dfs[lang_code]["bp_" + str(proportion)] = dfs[lang_code].apply(lambda x: get_disagreement_rating(x, "personas", allowed_disagree=proportion, include_base=True), axis=1)


    return dfs


def save_results_in_submission_format(results_dfs, method):
    """
    Docstring for save_results_in_submission_format
    
    :param results_dfs: Results dataframe, as calculated in get_results
    :param method: Aggregation method to use for the final results
    """

    Path(outputs_folder).mkdir(parents=True, exist_ok=True)

    for lang_code in lang_codes:
        df = results_dfs[lang_code][["id", method]]
        df = df.rename({method: "polarization"}, axis=1)

        df.to_csv(outputs_folder + f"/pred_{lang_code}.csv", index=False)
    
    return

def main():
    aggregated_dfs = aggregate(save=True)

    all_results = get_results(aggregated_dfs)

    # results are output in outputs/subtask1
    save_results_in_submission_format(all_results, "bp_0.1")

if __name__ == "__main__":
    main()