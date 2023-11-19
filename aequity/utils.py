import numpy as np
import pandas as pd

def generate_sample_sizes(max_sample_size : int = 5000, log_scale: int = 2, min_sample_size: int = 64, absolute_scale = None):
    sample_size_list = list()

    # print('Absolute scale', absolute_scale)
    if(absolute_scale == False):
        current_sample_size = min_sample_size
        while current_sample_size < max_sample_size:
            sample_size_list.append(current_sample_size)
            current_sample_size = current_sample_size * log_scale
    # if(absolute_scale == False):
    #     sample_size = int(max_sample_size)
    #     while sample_size > min_sample_size:
    #         sample_size_list.append(sample_size)
    #         sample_size = int(sample_size / log_scale)
    #     sample_size_list.append(min_sample_size)
    
    else:
        for sample_size in range(min_sample_size, max_sample_size, absolute_scale):
            sample_size_list.append(sample_size)
        sample_size_list.append(max_sample_size)
    sample_size_list.sort()
    # print(sample_size_list)
    return sample_size_list

def get_mcse_discrete(fcn_data, current_label, current_dataset, current_bootstrap):
    #print(fcn_data.dtypes)
    #print('Current_label', type(current_label))
    #print('dataset', type(current_dataset))
    #print('bootstrap', type(current_bootstrap))


    test = fcn_data[(fcn_data["label"] == current_label) & 
                    (fcn_data["dataset"] == current_dataset) &
                    (fcn_data["bootstrap"] == current_bootstrap)]
    test["log_sample_size"] = np.log2(test["sample_size"])
    test = test.groupby("log_sample_size").agg({'s_estimators': ['mean']})
    test = test.reset_index()


    mcse_index = np.argmax(np.diff(test["s_estimators"]["mean"], n=2)) + 2
    mcse_log_sample_size = test["log_sample_size"][mcse_index]
    return mcse_log_sample_size

def post_process_estimands(df:pd.DataFrame, 
    demographic:str, 
    n_bootstraps:int, 
    outcome_pre:str, 
    outcome_post:str, 
    demographic_pre:str, 
    demographic_post:str):
    """Method to post-process the estimands to measure bias. 

    Args:
        df (pd.DataFrame): The dataframe output by the mitigate_bias_customs function. 
        demographic (str): Demographic to evaluate on.
        n_bootstraps (int): Number of bootstraps. 
        outcome_pre (str): Outcome selected for pre-mitigation. 
        outcome_post (str): Outcome selected for post-mitigation. 
        demographic_pre (str): Demographic trained on pre-mitigation. 
        demographic_post (str): Demographic trained on post mitigation. 

    Returns:
        pd.DataFrame: Effect of intervention on metric. 
    """
    df2 = df[
        (df['demographics_test'] == demographic) &
        (df['sample_size'] == df['sample_size'].max())
    ]


    df2 = df2.groupby(["sample_size", "demographics_train", "outcome_train"]).agg({
            "estimands": ["mean", "std"]
    })
    df2["auc_mean"] = df2["estimands"]["mean"]
    df2["auc_se"]   = df2["estimands"]["std"] / n_bootstraps

    df2.reset_index(inplace=True)
    df2.drop(["estimands", "sample_size"], axis=1, inplace=True)

    pre =  df2[
        ((df2['demographics_train'] == demographic_pre)  & (df2['outcome_train'] == outcome_pre))]
    pre['intervention'] = 'Pre'
    post =  df2[
        ((df2['demographics_train'] == demographic_post)  & (df2['outcome_train'] == outcome_post))]
    post['intervention'] = 'Post'

    df3 = pd.concat([pre, post], axis=0)
    df3 = df3[['auc_mean', 'auc_se', 'intervention']]
    return df3