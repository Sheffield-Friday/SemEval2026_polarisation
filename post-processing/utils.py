import pandas as pd


lang_codes =    ['amh', 'arb', 'ben', 'deu', 'eng', 'fas', 'hau', 'hin', 'ita', 'khm', 'mya', 'nep', 'ori', 'pan', 'pol', 
                'rus', 'spa', 'swa', 'tel', 'tur', 'urd', 'zho']
lang_codes_subtask3 = ['amh', 'arb', 'ben', 'deu', 'eng', 'fas', 'hau', 'hin', 'khm', 'nep', 'ori', 'pan', 'spa', 'swa', 'tel', 'tur', 'urd', 'zho']

# I have assumed that the annotations are in this order, it's the one listed on the codabench competition!
categories_subtask_2 = ["political","racial/ethnic","religious","gender/sexual","other"]

def get_counts(row, model, annotator, yes_output_col="yes", no_output_col="no"):
    """
    Docstring for get_counts
    
    :param row: Row from a combined resulta dataframe
    :param model (list or str): Models to include in the count
    :param annotator (list or str): Annotators to include in the count. By default includes all annotators
    """

    # get the correct model and annotators to generate the column names
    if type(model) == str:
        model_cols = [model]
    else:
        model_cols = model
    
    if type(annotator) == str:
        ann_cols = [annotator]
    else:
        ann_cols = annotator
    
    # iterate over the eligible columns and increment the yes/no counts
    counts = {'yes': 0, 'no': 0}
    found_filtered = False

    for m in model_cols:
        for a in ann_cols:
            col_name = m + a

            if col_name in row.index:
                if row[col_name] == "FILTERED":
                    found_filtered = True
                    counts["no"] += 1
                else:
                    counts[row[col_name]] += 1

    if found_filtered:
        # row has been filtered out - treat all annotations as if they were 'no'
        row[yes_output_col] = 0
        row[no_output_col] = counts["no"] + counts["yes"]
    else:
        row[yes_output_col] = counts["yes"]
        row[no_output_col] = counts["no"]
    return row

def get_counts_for_category(row, model, annotator, yes_output_col="yes", no_output_col="no", categories=categories_subtask_2):
    """
    Docstring for get_counts
    
    :param row: Row from a combined resulta dataframe
    :param model (list or str): Models to include in the count
    :param annotator (list or str): Annotators to include in the count. By default includes all annotators
    """
    # print("CALLING GET_COUNTS")
    # print(row)

    # get the correct model and annotators to generate the column names
    if type(model) == str:
        model_cols = [model]
    else:
        model_cols = model
    
    if type(annotator) == str:
        ann_cols = [annotator]
    else:
        ann_cols = annotator
    
    if type(categories) == str:
        cat_cols = [categories]
    else:
        cat_cols = categories
    
    # iterate over the eligible columns and increment the yes/no counts
    counts = {'yes': 0, 'no': 0}
    found_filtered = False

    for c in cat_cols:
        for m in model_cols:
            for a in ann_cols:
                col_name = m + a + "_" + c
                # print(row['id'], col_name, row[col_name])

                if col_name in row.index:
                    if row[col_name] == "FILTERED":
                        counts["no"] += 1
                    else:
                        counts[row[col_name].lower()] += 1

    # check for 'FILTERED' here - in case of base anntoators only the conditional on line 93 will never be triggered
    if "FILTERED" in row.values:
        # row has been filtered out - treat all annotations as if they were 'no'
        row[yes_output_col] = 0
        row[no_output_col] = counts["no"] + counts["yes"]
    else:
        row[yes_output_col] = counts["yes"]
        row[no_output_col] = counts["no"]
    
    return row

def validate_df(df, models, annotators, categories=[""]):
    """Checks that all of the expected columns are present (i.e. columns for each model and annotator)"""
    if not categories:
        categories = [""]

    missing_cols = []

    for model in models:
        for annotator in annotators:
            for category in categories:
                if category:
                    col_name = model + annotator + "_" + category
                else:
                    col_name = model + annotator

                if col_name not in df.columns:
                    # Meta-Llama-3-8B-Instruct is only annotated with base
                    if model == 'Meta-Llama-3-8B-Instruct-' and annotator != "base":
                        pass
                    else:
                        missing_cols.append(col_name)
    
    if len(missing_cols):
        print("Missing columns: ")
        print(*missing_cols, sep=", ")
        return False
    
    return True

def load_aggregated_dfs(aggregated_filepath):
    dfs = {}

    for lang_code in lang_codes:
        dfs[lang_code] = pd.read_csv(aggregated_filepath.replace('LANG_CODE', lang_code))
    
    return dfs

def get_disagreement_rating(row, col_suffix, allowed_disagree=0, include_base=False, col_category = ""):
    """
    Docstring for get_final_rating
    
    :param allowed_disagree (int or float): Number/proportion of annotators that can differ from the majority before the content is deemed 'polarising'
    :param include_base (bool): Whether to include the base results (yes_base (aka is polarised) and no_base (aka is not polarised
                                If True, yes_base is added to the minority count to show disagreement.
    """
    if col_category:
        col_category += "_"

    counts = [row[col_category + "no_" + col_suffix], row[col_category + "yes_" + col_suffix]]

    majority = counts.index(max(counts))
    minority = 1 if majority == 0 else 0

    if type(allowed_disagree) == int:
        if include_base:
            polarised = counts[minority] + row[col_category + "yes_base"] > allowed_disagree
        else:
            polarised = counts[minority] > allowed_disagree

    elif type(allowed_disagree) == float:
        if include_base:
            polarised = ((counts[minority] + row[col_category + "yes_base"]) / (sum(counts) + row[col_category + "yes_base"] + row[col_category + "no_base"])) > allowed_disagree
        else:
            polarised = (counts[minority] / sum(counts)) > allowed_disagree

    if polarised:
        return 1
    else:
        return 0 

def get_majority_rating(row, col_suffix, col_category=""):
    if col_category:
        col_category += "_"
    yes_count = row[col_category + "yes_" + col_suffix]
    no_count = row[col_category + "no_" + col_suffix]

    if yes_count >= no_count:
        return 1
    else:
        return 0
    
def split_dict_into_cols(row, colname, categories):
    annotation = eval(row[colname])
    for i in range(0, len(categories)):
        row[colname + "_" + categories[i]] = annotation[i]
    
    return row
