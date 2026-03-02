from pathlib import Path
import pandas as pd
import json
from collections import Counter
from sklearn.metrics import f1_score
import seaborn
import matplotlib.pyplot as plt

from utils import *

raw_results_folder = {"test": "annotations/st2/test/", "dev": "annotations/st2/dev/", "dev_filtered": "annotations/st2/dev_filtered/"}
subtask1_dev_folder = "./outputs/subtask1_dev_bp_0.1/"
outputs_folder = "outputs/subtask_2"
aggregated_results_folder = "outputs/subtask2_aggregated_annotations"
models = ["gemma-3-27b-it-", "llama-3-70b-", "qwen25-72b-"]
annotators = {"test": ["base", "hate-personas-annotator_0", "hate-personas-annotator_1", "hate-personas-annotator_2", "hate-personas-annotator_3", "hate-personas-annotator_4",
              "hate-personas-annotator_5", "hate-personas-annotator_6", "hate-personas-annotator_7", "hate-personas-annotator_8", "hate-personas-annotator_9"],
              "dev": ["base", "hate-personas-annotator_0", "hate-personas-annotator_1", "hate-personas-annotator_2", "hate-personas-annotator_3", "hate-personas-annotator_4",
              "hate-personas-annotator_5", "hate-personas-annotator_6", "hate-personas-annotator_7", "hate-personas-annotator_8", "hate-personas-annotator_9",
              "pol-personas-annotator_0", "pol-personas-annotator_1", "pol-personas-annotator_2", "pol-personas-annotator_3", "pol-personas-annotator_4",
              "pol-personas-annotator_5", "pol-personas-annotator_6", "pol-personas-annotator_7", "pol-personas-annotator_8", "pol-personas-annotator_9"],
              "dev_filtered": ["base", "hate-personas-annotator_0", "hate-personas-annotator_1", "hate-personas-annotator_2", "hate-personas-annotator_3", "hate-personas-annotator_4",
              "hate-personas-annotator_5", "hate-personas-annotator_6", "hate-personas-annotator_7", "hate-personas-annotator_8", "hate-personas-annotator_9",
              "pol-personas-annotator_0", "pol-personas-annotator_1", "pol-personas-annotator_2", "pol-personas-annotator_3", "pol-personas-annotator_4",
              "pol-personas-annotator_5", "pol-personas-annotator_6", "pol-personas-annotator_7", "pol-personas-annotator_8", "pol-personas-annotator_9"]}

gold_filepath = "./data/subtask2/dev_gold/LANG_CODE.csv"

def simulate_dev_filtered_out_from_subtask1(save=False):
    """
    Generates results to simulate the dev set being run on the 'polarised' data only. Don't use - results in really slow processing later.
    
    :param save: Whether to save the results
    """
    for lang_code in lang_codes:
        subtask1 = pd.read_csv(f"{subtask1_dev_folder}pred_{lang_code}.csv")
        subtask1_is_polarised = subtask1.loc[subtask1['polarization'] == 1]

        lang_folder = Path(raw_results_folder["dev"] + lang_code + "/")
        Path(f"{raw_results_folder['dev_filtered']}/{lang_code}/").mkdir(parents=True, exist_ok=True)

        for filepath in lang_folder.glob("*.csv"):
            filename = str(filepath).split("/")[-1]
            subtask2_file = pd.read_csv(filepath)

            if "base" not in filename:
                subtask2_file = subtask2_file[subtask2_file['id'].isin(subtask1_is_polarised['id'])]

            
            subtask2_file.to_csv(f"{raw_results_folder['dev_filtered']}/{lang_code}/{filename}", index=False)
    return


def aggregate(nan_processing="equalsno", save=False, split="test"):
    """
    Returns dictionary of {lang_code : aggregated_df} 

    :param nan_processing:  If "remove", discards the rows that have base annotatiosn but no persona annotations. 
                            If "equalsno", sets all persona annotations for filtered out rows to 'no'. 
                            If "allequalsno", sets ALL annotations for filtered out rows to 'no'
    :param save: If True, saves the data

    Each aggregated df has the following columns:
    - id
    - base_CATEGORY_yes - number of base models who annotated 'yes' for category CATEGORY
    - base_CATEGORY_no - number of base models who annotated 'no' for category CATEGORY
    """

    aggregated_dfs = {}

    if save:
        Path(aggregated_results_folder).mkdir(parents=True, exist_ok=True)
    
    for lang_code in lang_codes:
        combined_df = -1
        lang_folder = Path(raw_results_folder[split] + lang_code + "/")
        for filepath in lang_folder.glob("*.csv"):
            if type(combined_df) == int:
                combined_df = pd.read_csv(filepath).drop(columns=["language"])
                # all_annotations_length = len(combined_df)
                # first_df_filename = filepath

                combined_df = combined_df.apply(lambda x: split_dict_into_cols(x, colname=combined_df.columns[-1],
                                                                                categories=categories_subtask_2), axis=1)
            else:
                temp = pd.read_csv(filepath).drop(columns=["language"])
                temp = temp.apply(lambda x: split_dict_into_cols(x, colname=temp.columns[-1], categories=categories_subtask_2), axis=1)

                # if len(temp) != all_annotations_length:
                #     print(f"ERROR: Difference in dataframe length for {first_df_filename} ({all_annotations_length}) and {filepath} ({len(temp)})!")
                #     exit(1)
                combined_df = combined_df.merge(temp, on="id", how="left")
        
        # validate
        is_valid = validate_df(combined_df, models=models, annotators=annotators[split], categories=categories_subtask_2)
        if is_valid:
            print(f"{lang_code} results validated")
        else:
            print(f"{lang_code} invalid!")
            exit()

        # print(combined_df.columns)

        # REPLACE NAN WITH 'NO' ANNOTATION OR REMOVE NAN ROWS ENTIRELY
        # the na values come from some non-polarised data being filtered out via subtask 1
        if nan_processing == "remove":
            combined_df = combined_df.dropna(axis=0, how="any")
        elif nan_processing == "equalsno":
            combined_df = combined_df.fillna("no")
        elif nan_processing == "ALLequalsno":
            combined_df = combined_df.fillna("FILTERED")

            # Filtered out rows do NOT have all nos for the base annotations - there's an error somewhere

            # filtered_out_rows = combined_df[combined_df["llama-3-70b-pol-personas-annotator_0"] == "FILTERED"]

            # for base_annotator_col in [model + "base" for model in models]:
            #     print(filtered_out_rows[base_annotator_col].value_counts())
        else:
            print(f"Invalid nan_processing parameter {nan_processing}!")
            exit()
        
        # aggregate the annotators
        if split == "dev" or split == "dev_filtered":
            output_df_cols_base = ["yes_base", "no_base", "yes_personas", "no_personas", "yes_personas_pol", "no_personas_pol", "yes_personas_ALL", "no_personas_ALL"]
        else:
            output_df_cols_base = ["yes_base", "no_base", "yes_personas", "no_personas"]
            
        full_output_cols = ["id"]
        for category in categories_subtask_2:
            
            full_output_cols.extend([category + "_" + i for i in output_df_cols_base])

            # all models with the base annotator ONLY
            combined_df = combined_df.apply(lambda x: get_counts_for_category(x, model=models, annotator="base", 
                                                yes_output_col = category + "_yes_base", no_output_col = category + "_no_base", categories=category), axis=1)
            
            # all models with standard (hate) persona annotators ONLY
            combined_df = combined_df.apply(lambda x: get_counts_for_category(x, model=models, annotator=annotators[split][1:11], 
                                            yes_output_col = category + "_yes_personas", no_output_col = category + "_no_personas", categories=category), axis=1)
            
            if split == "dev" or split == "dev_filtered":                
                # all models with persona pol anntoators ONLY
                combined_df = combined_df.apply(lambda x: get_counts_for_category(x, model=models, annotator=annotators[split][12:], 
                                                yes_output_col = category + "_yes_personas_pol", no_output_col = category + "_no_personas_pol", categories=category), axis=1)
                
                # hate and pol personas 
                combined_df = combined_df.apply(lambda x: get_counts_for_category(x, model=models, annotator=annotators[split][1:], 
                                                yes_output_col = category + "_yes_personas_ALL", no_output_col = category + "_no_personas_ALL", categories=category), axis=1)


        aggregated_dfs[lang_code] = combined_df[full_output_cols]
        # print(aggregated_dfs[lang_code])

        if save:
            print("SAVING")
            print(aggregated_results_folder + f"/{lang_code}.csv")
            aggregated_dfs[lang_code].to_csv(aggregated_results_folder + f"/{lang_code}.csv", index=False)

    return aggregated_dfs


def get_macro_f1(results_df, prefix, lang_code):
    gold_labels = pd.read_csv(gold_filepath.replace("LANG_CODE", lang_code))

    results_cols = [prefix + "_" + i for i in categories_subtask_2]
    print(results_cols)

    if results_df['id'].equals(gold_labels['id']):
        return f1_score(gold_labels[categories_subtask_2], results_df[results_cols], average='macro')
    else:
        gold_labels = gold_labels[gold_labels['id'].isin(results_df['id'])]

        results_df = results_df.set_index('id').sort_index()
        gold_labels = gold_labels.set_index('id').sort_index()

        return f1_score(gold_labels[categories_subtask_2], results_df[results_cols], average='macro')

def get_results(dfs=None, split="test", override_filename=""):
    """
    
    :param dfs: Aggregated dataframes as calculated in aggregate()

    Returns dictionary of {lang_code : results_df} 

    """
    if not dfs:
        dfs = load_aggregated_dfs(aggregated_results_folder + "/LANG_CODE.csv")

    f1_scores = []

    for lang_code in lang_codes:
        for category in categories_subtask_2:
            dfs[lang_code]["base_maj_" + category] = dfs[lang_code].apply(lambda x: get_majority_rating(x, "base", col_category=category), axis=1)


            for proportion in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
                dfs[lang_code]["personas_" + str(proportion) + "_" + category] = dfs[lang_code].apply(lambda x: get_disagreement_rating(x, "personas", allowed_disagree=proportion, col_category=category), axis=1)

                dfs[lang_code]["bp_" + str(proportion)+ "_" + category] = dfs[lang_code].apply(lambda x: get_disagreement_rating(x, "personas", allowed_disagree=proportion, include_base=True, col_category=category), axis=1)

                if split == "dev" or split == "dev_filtered":

                    dfs[lang_code]["personas_pol_" + str(proportion) + "_" + category] = dfs[lang_code].apply(lambda x: get_disagreement_rating(x, "personas_pol", allowed_disagree=proportion, col_category=category), axis=1)
                    
                    dfs[lang_code]["personas_ALL_" + str(proportion) + "_" + category] = dfs[lang_code].apply(lambda x: get_disagreement_rating(x, "personas_ALL", allowed_disagree=proportion, col_category=category), axis=1)

                    dfs[lang_code]["bpol_" + str(proportion)+ "_" + category] = dfs[lang_code].apply(lambda x: get_disagreement_rating(x, "personas_pol", allowed_disagree=proportion, include_base=True, col_category=category), axis=1)
                    
                    dfs[lang_code]["bALL_" + str(proportion)+ "_" + category] = dfs[lang_code].apply(lambda x: get_disagreement_rating(x, "personas_ALL", allowed_disagree=proportion, include_base=True, col_category=category), axis=1)

        if split == "dev" or split == "dev_filtered":
            f1_scores.append({"lang": lang_code, "method": "base_majority", 
                        "macro-f1": get_macro_f1(dfs[lang_code], "base_maj", lang_code)})
            
            for proportion in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
                f1_scores.append({"lang": lang_code, "method": "personas_" + str(proportion), 
                                "macro-f1": get_macro_f1(dfs[lang_code], "personas_" + str(proportion), lang_code)})
                        
                f1_scores.append({"lang": lang_code, "method": "bp_" + str(proportion), 
                                "macro-f1": get_macro_f1(dfs[lang_code], "bp_" + str(proportion), lang_code)})
                
                f1_scores.append({"lang": lang_code, "method": "personas_pol_" + str(proportion), 
                                "macro-f1": get_macro_f1(dfs[lang_code], "personas_pol_" + str(proportion), lang_code)})
                
                f1_scores.append({"lang": lang_code, "method": "personas_ALL_" + str(proportion), 
                                "macro-f1": get_macro_f1(dfs[lang_code], "personas_ALL_" + str(proportion), lang_code)})
                
                f1_scores.append({"lang": lang_code, "method": "bpol_" + str(proportion), 
                                "macro-f1": get_macro_f1(dfs[lang_code], "bpol_" + str(proportion), lang_code)})
                
                f1_scores.append({"lang": lang_code, "method": "bALL_" + str(proportion), 
                                "macro-f1": get_macro_f1(dfs[lang_code], "bALL_" + str(proportion), lang_code)})
    if split == "dev" or split == "dev_filtered":      
        results_df = pd.DataFrame(f1_scores)

        if override_filename:
            results_df.to_csv(f"./outputs/{override_filename}", index=False)
        else:
            results_df.to_csv(f"./outputs/{split}_results_subtask_2.csv", index=False)
        return results_df
    else:
        return dfs

def get_dataframe_lengths():
    
    for lang_code in lang_codes:
        dframe_lengths = []
        lang_folder = Path(raw_results_folder + lang_code + "/")
        for filepath in lang_folder.glob("*.csv"):
            dframe_lengths.append(len(pd.read_csv(filepath)))
        
        print(f"\n\n{lang_code}:")
        print(Counter(dframe_lengths))

def save_results_in_submission_format(results_dfs, method):
    """
    Docstring for save_results_in_submission_format
    
    :param results_dfs: Results dataframe, as calculated in get_results
    :param method: Aggregation method to use for the final results
    """

    Path(outputs_folder).mkdir(parents=True, exist_ok=True)
    output_cols = [method + "_" + i for i in categories_subtask_2]

    renamed_cols = dict(zip(output_cols, categories_subtask_2))

    output_cols = ["id"] + output_cols

    print(renamed_cols)

    for lang_code in lang_codes:
        df = results_dfs[lang_code][output_cols]
        df = df.rename(renamed_cols, axis=1)

        df.to_csv(outputs_folder + f"/pred_{lang_code}.csv", index=False)
    
    return


def visualise_dev_set_results(dev_set_results, filtered=False, file_suffix=""):
    # heatmap - macro f1 per method and language
    dev_set_per_method = dev_set_results.pivot_table(
                        index='lang',                       # rows of the heatmap
                        columns='method',                   # columns of the heatmap
                        values="macro-f1",                       # colour intensity
                    )
    plt.figure() 

    seaborn.heatmap(dev_set_per_method, cbar_kws={"label": "Macro-F1"}, cmap="Reds")

    plt.ylabel("Language", style="italic")
    plt.xlabel("Method", style="italic")

    if filtered:
        out_filename = f"./outputs/f1_dev_set_subtask_2_filtered{file_suffix}.png" 
    else:
        out_filename = f"./outputs/f1_dev_set_subtask_2{file_suffix}.png" 

    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    plt.close()

    # most effective methods
    best_method_per_language = dev_set_results.loc[dev_set_results.groupby('lang')['macro-f1'].idxmax()]
    best_method_per_language_vc = best_method_per_language['method'].value_counts()
    print(best_method_per_language_vc)

    plt.figure() 

    best_method_per_language_vc.plot(kind='bar', color='red', edgecolor='black')

    plt.ylabel("Number of Languages", style="italic")
    plt.xlabel("Method with Highest Macro-F1", style="italic")
    
    if filtered:
        out_filename = f"./outputs/best_methods_subtask_2_filtered{file_suffix}.png"
    else:
         out_filename = f"./outputs/best_methods_subtask_2{file_suffix}.png"
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # DEV SET
    # simulate_dev_filtered_out_from_subtask1(save=True)
    # aggregated_dfs_1 = aggregate(save=False, nan_processing="equalsno", split="dev_filtered")
    # dev_set_results_1 = get_results(aggregated_dfs_1, split="dev_filtered")     
    # visualise_dev_set_results(dev_set_results_1, filtered=True)
    
    # aggregated_dfs_2 = aggregate(save=False, nan_processing="ALLequalsno", split="dev_filtered")
    # dev_set_results_2 = get_results(aggregated_dfs_2, split="dev_filtered", override_filename="dev_filtered_results_subtask_2_filtered_ALLequalsno.csv")    
    # assert not aggregated_dfs_1["amh"].equals(aggregated_dfs_2["amh"])
    # assert not dev_set_results_1.equals(dev_set_results_2)
    # visualise_dev_set_results(dev_set_results_2, filtered=True, file_suffix="_ALLequalsno")
    
    # aggregated_dfs_3 = aggregate(save=False, nan_processing="equalsno", split="dev")
    # dev_set_results_3 = get_results(aggregated_dfs_3, split="dev")    
    # assert not aggregated_dfs_1["amh"].equals(aggregated_dfs_3["amh"])
    # assert not dev_set_results_3.equals(dev_set_results_2)
    # visualise_dev_set_results(dev_set_results_3, filtered=False)
    # exit()

    # TEST SET
    aggregated_dfs = aggregate(save=True, nan_processing="equalsno")    # equalsno will have no effect on base_maj, so subtask 2 results are unfiltered by subtask 1 results
    results_dfs = get_results(aggregated_dfs)
    save_results_in_submission_format(results_dfs, "base_maj")

if __name__ == "__main__":
    main()