import itertools
import argparse
import yaml
import numpy as np
import pandas as pd

def get_mcse(fcn_data, current_label, current_dataset, current_bootstrap):
    test = fcn_data[(fcn_data["label"] == current_label) & 
                    (fcn_data["dataset"] == current_dataset) &
                    (fcn_data["bootstrap"] == current_bootstrap)].copy()
    test["log_sample_size"] = np.log2(test["sample_size"])
    test = test.groupby("log_sample_size").agg({'s_estimators': ['mean']})
    test = test.reset_index()
    mcse_index = np.argmax(np.diff(test["s_estimators"]["mean"], n=2)) + 2
    mcse_log_sample_size = test["log_sample_size"][mcse_index]
    return mcse_log_sample_size

def main(measure_disparity_filepath:str, outfile_path:str):
    fcn_data = pd.read_csv(measure_disparity_filepath, sep="\t")
    datasets = fcn_data["dataset"].drop_duplicates().to_list()
    labels = fcn_data["label"].drop_duplicates().to_list()
    bootstraps = fcn_data["bootstrap"].drop_duplicates().to_list()

    datasets_vector = []
    labels_vector = []
    bootstraps_vector = []
    mcse_vector = []

    for value, item, version in itertools.product(labels, datasets, bootstraps):
        datasets_vector.append(item)
        labels_vector.append(value)
        bootstraps_vector.append(version)
        mcse_vector.append(get_mcse(fcn_data, value, item, version))
    
    metadata = pd.DataFrame({
        "dataset": datasets_vector,
        "label": labels_vector,
        "bootstrap": bootstraps_vector,
        "mcse": mcse_vector
    })

    vc = metadata.groupby(["dataset", "label"]).agg({
        "mcse": ["mean", "std"]
    })
    vc["mean_mcse"] = vc["mcse"]["mean"]
    vc["sd_mcse"] = vc["mcse"]["std"] / 50
    vc.drop(["mcse"], axis=1, inplace=True)
    vc.reset_index(inplace=True)
    vc[["step_1", "Training", "step_3", "Testing", "demographic"]] = vc["dataset"].str.split("_", expand=True, n = 4)
    vc.drop(["step_1", "step_3"], axis=1, inplace=True)
    vc.to_csv(outfile_path, sep="\t")


def fit(vc, group:str, label:str):
    """Fit and perform model. 

    Args:
        vc (_type_): _description_
        group (str): _description_
        label (str): _description_
    """

    # Subset to relevant demographics and labels. 
    vc_subset_demographics_label = vc[
        (vc['demographic'] == group) &
        (vc['label'] == label)
    ]

    # Subset to minimum AEq value. 
    vc_min_aeq = vc_subset_demographics_label[
        vc_subset_demographics_label['mean_mcse'] == vc_subset_demographics_label['mean_mcse'].min()
    ]
    optimal_dataset = vc_min_aeq['dataset']

    # Select model with optimal dataset. 
    optimal_model = self.fcn_data[
        fcn_data['dataset'] == optimal_dataset
    ]

    return optimal_model

def predict(predictions_path:str, group:str, label:str):
    predictions_df = pd.read_csv(predictions_path, sep="\t")
    optimal_model = self.fit(self.vc, group=group, label=label)

    predictions_df = predictions_df[
        predictions_df['dataset'] == optimal_model
    ]

    return predictions_df


    






    



    pass


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration arguments for convergence sample size estimtation.')
    parser.add_argument("--measure_disparity_filepath", required=True, type=str,
                        help="File path to measure disparities.")
    config_kwargs = parser.parse_args()
    main(**(config_kwargs))
