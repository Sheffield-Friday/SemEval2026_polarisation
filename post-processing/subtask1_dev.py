import pandas as pd
from sklearn.metrics import f1_score
import seaborn
import matplotlib.pyplot as plt
from pathlib import Path

from utils import *

filepath = "./combined/LANG_CODE.csv"
aggregated_filepath = "./combined/aggregated/LANG_CODE.csv"
gold_filepaths = ["./data/subtask1/dev_gold/LANG_CODE.csv", "./data/subtask2/dev_gold/LANG_CODE.csv", "./data/subtask3/dev_gold/LANG_CODE.csv"]
models = ['gemma-3-27b-it-', 'llama-3-70b-', 'Meta-Llama-3-8B-Instruct-', 'qwen25-72b-']
annotators =    ['base', 'annotator_0', 'annotator_1', 'annotator_2', 'annotator_3', 'annotator_4', 'annotator_5', 
                'annotator_6', 'annotator_7', 'annotator_8', 'annotator_9']




def aggregate_counts(save=True):
    aggregated_dfs = {}

    for lang_code in lang_codes:
        df = pd.read_csv(filepath.replace('LANG_CODE', lang_code))

        valid_df = validate_df(df, models=models, annotators=annotators)
        if valid_df:
            print("Dataframe validated")
        else:
            print(f"Columns missing for lang code {lang_code}! Skipping")
            continue
        

        # AGGREGATED COUNTS
        output_df_cols = ["id", "yes_all", "no_all", "yes_base", "no_base", "yes_personas", "no_personas"]

        # all models/annotators
        df = df.apply(lambda x: get_counts(x, model=models, annotator=annotators, 
                                           yes_output_col="yes_all", no_output_col="no_all"), axis=1)

        # all models with the base annotator ONLY
        df = df.apply(lambda x: get_counts(x, model=models, annotator="base", 
                                            yes_output_col="yes_base", no_output_col="no_base"), axis=1)
        
        # all models with persona annotators ONLY
        df = df.apply(lambda x: get_counts(x, model=models, annotator=annotators[1:], 
                                           yes_output_col="yes_personas", no_output_col="no_personas"), axis=1)
        
        # per model
        for model in models:

            # Meta-Llama-3-8B-Instruct- only used with base annotator; can't aggregate with 1 annotator so skip
            if model != 'Meta-Llama-3-8B-Instruct-':
                # all annotators
                df = df.apply(lambda x: get_counts(x, model=model, annotator=models, 
                                                yes_output_col="yes_" + model[:-1], no_output_col="no_" + model[:-1]), axis=1)

                # persona annotators only
                df = df.apply(lambda x: get_counts(x, model=model, annotator=annotators[1:], 
                                                yes_output_col="yes_" + model[:-1] + "_personas", no_output_col="no_" + 
                                                model[:-1] + "_personas"), axis=1)

                output_df_cols.extend(["yes_" + model[:-1], "no_" + model[:-1], "yes_" + model[:-1] + "_personas", "no_" + model[:-1] + "_personas"])

        # per annotator - not used because each model uses a different set of annotators, so each annotator_0, annotator_1 etc is different
        # for annotator in annotators:
        #     df = df.apply(lambda x: get_counts(x, model=models, annotator=annotator, 
        #                                        yes_output_col="yes_" + annotator, no_output_col="no_" + annotator), axis=1)
            
        #     output_df_cols.extend(["yes_" + annotator, "no_" + annotator])
        
        aggregated_dfs[lang_code] = df[output_df_cols]

        if save:
            aggregated_dfs[lang_code].to_csv(f"./combined/aggregated/{lang_code}.csv")
    
    return aggregated_dfs

def get_macro_f1(results_df, results_col, lang_code, subtask=1):
    gold_labels = pd.read_csv(gold_filepaths[subtask-1].replace("LANG_CODE", lang_code))

    assert results_df['id'].equals(gold_labels['id'])

    return f1_score(gold_labels['polarization'], results_df[results_col], average='macro')

def get_dev_set_results(calculate_aggregated=False):
    """
    Calculates the macro-f1 scores for each language and method, saving them in outputs/
    
    :param calculate_aggregated: If False, uses the existing aggregated data in combined/aggregated, instead of recalculating it.
    """
    if calculate_aggregated:
        dfs = aggregate_counts(save=False)
    else:
        dfs = load_aggregated_dfs(aggregated_filepath=aggregated_filepath)

    f1_scores = []

    for lang_code in lang_codes:
        dfs[lang_code]["base_maj"] = dfs[lang_code].apply(lambda x: get_majority_rating(x, "base"), axis=1)
        f1_scores.append({"lang": lang_code, "method": "base_majority", 
                          "macro-f1": get_macro_f1(dfs[lang_code], "base_maj", lang_code)})

        for proportion in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            dfs[lang_code]["personas_" + str(proportion)] = dfs[lang_code].apply(lambda x: get_disagreement_rating(x, "personas", allowed_disagree=proportion), axis=1)
            f1_scores.append({"lang": lang_code, "method": "personas_" + str(proportion), 
                              "macro-f1": get_macro_f1(dfs[lang_code], "personas_" + str(proportion), lang_code)})

            dfs[lang_code]["bp_" + str(proportion)] = dfs[lang_code].apply(lambda x: get_disagreement_rating(x, "personas", allowed_disagree=proportion, include_base=True), axis=1)
            f1_scores.append({"lang": lang_code, "method": "bp_" + str(proportion), 
                            "macro-f1": get_macro_f1(dfs[lang_code], "bp_" + str(proportion), lang_code)})
        # print(dfs[lang_code].columns)

        # base_maj_df = dfs[lang_code][["id", "base_maj"]]
        # base_maj_df = base_maj_df.rename(columns={"base_maj": "polarization"})
        # base_maj_df.to_csv(final_pred_base_filepath.replace("LANG_CODE", lang_code), index=False)

        # base_maj_df = dfs[lang_code][["id", "personas_maj"]]
        # base_maj_df = base_maj_df.rename(columns={"personas_maj": "polarization"})
        # base_maj_df.to_csv(final_pred_persona_filepath.replace("LANG_CODE", lang_code), index=False)

    results_df = pd.DataFrame(f1_scores)

    results_df.to_csv("./outputs/dev_results.csv", index=False)

    return results_df, dfs

def save_results_in_submission_format(results_dfs, outputs_folder, method):
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

def visualise_dev_set_results(dev_set_results):
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
    
    out_filename = f"./outputs/f1_dev_set_subtask_1.png" 
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
    
    out_filename = f"./outputs/best_methods_subtask1.png" 
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    dev_set_results, full_results = get_dev_set_results()

    visualise_dev_set_results(dev_set_results)

    save_results_in_submission_format(full_results, "./outputs/subtask1_dev_bp_0.1/", method="bp_0.1")


if __name__ == "__main__":
    main()