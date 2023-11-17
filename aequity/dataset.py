import torch
import os
import numpy as np
import pandas as pd 

from os.path import exists
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, random_split, ConcatDataset, DataLoader

def weighted_sampler(subset, mode:str, label_map:dict=None):
    label_map = {
        0 : ["None", 1], 
        1 : ["Pneumonia", 1],
        2 : ["Pneumothorax", 2],
        3 : ["Atelectasis", 0.5],
        4 : ["Cardiomegaly", 0.5],
        5 : ["Consolidation", 2],
        6 : ["Edema", 1],
        7 : ["Enlarged CM", 2],
        8 : ["Opacity", 0.5],
        9 : ["Effusion", 0.5]
    }

    if(mode == "frequency"):
        #y_train_indices = subset.indices
        #all_dataset = subset.dataset

        y_train = get_labels(subset)
        print(y_train)
        unique_labels = np.unique(y_train)
        class_sample_count = np.array(
            [len(np.where(y_train == t)[0]) for t in unique_labels])
        print(class_sample_count)
        weights = 1. / class_sample_count
        weight_dict = {unique_label : weight for unique_label, weight in zip(unique_labels, weights)}
        print(weight_dict)
        samples_weight = np.array([weight_dict[t] for t in y_train])
        print(samples_weight)
        samples_weight = torch.from_numpy(samples_weight)

    if(mode == "mcse"):
        #y_train_indices = subset.indices
        y_train = get_labels(subset)
        samples_weight = np.array([label_map[t][1] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
    
    if(mode == "joint"):
        y_train = get_labels(subset)
        unique_labels = np.unique(y_train)
        class_sample_count = np.array(
            [len(np.where(y_train == t)[0]) for t in unique_labels])
        print(class_sample_count)
        weights = 1. / class_sample_count
        weight_dict = {unique_label : weight for unique_label, weight in zip(unique_labels, weights)}
        samples_weight = np.array([label_map[t][1] * weight_dict[t]  for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)


    return WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    
class CustomDataset(Dataset):
    """Dataset for Dissecting Datsaet bias

    Args:
        Dataset: Torch dataset type. 
    """
    def __init__(self, metadata_path, transform=None, target_transform=None):
        self.metadata_path = metadata_path
        self.metadata_df = pd.read_csv(metadata_path, sep="\t")
        self.targets = torch.from_numpy(np.array(self.metadata_df['target']))
        self.inputs = self.metadata_df.drop(['target'], axis=1)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.metadata_df.shape[0]
    
    def __getitem__(self, idx):
        predictors = torch.Tensor(self.inputs.iloc[idx])
        outcomes = self.targets[idx]

        if(self.transform):
            predictors = self.transform(predictors)
        
        if(self.target_transform):
            outcomes = self.target_transform(outcomes)
        
        return predictors, outcomes

def generate_metadata_file(
    root_dir:str,
    dataset_label:str,
    dataset_df:pd.DataFrame, 
    outcome_col:str, 
    outcome_cols:list,
    demographic_col:str,
    exclude_cols:list,
    demographic:str=None,
    threshold:float=0.5
    ):

    # Subset by demographic and outcome. 
    if(demographic != None):
        subset_df = dataset_df[
            (dataset_df[demographic_col] == demographic)
        ]   
    else:
        subset_df = dataset_df

    dataset_df['target'] = (dataset_df[outcome_col] > dataset_df[outcome_col].quantile(q=threshold)).apply(int)
    outcome_values = dataset_df['target']

    # Drop irrelevant variables.
    subset_df = subset_df.drop(
        outcome_cols + [demographic_col] + exclude_cols, axis=1
    )

    # Scale variables. 
    X_df_column_names = list(subset_df.columns)
    scaler = MinMaxScaler()
    X_vars = scaler.fit_transform(subset_df.to_numpy())
    X_df = pd.DataFrame(X_vars, columns=X_df_column_names)
    
    # Re-insert targets. 
    X_df['target'] = outcome_values

    # Output path. 
    out_path = os.path.join(root_dir, dataset_label + '.tsv')
    if(exists(out_path) == False): 
        X_df.to_csv(out_path, sep="\t", index=False)
    
    return out_path


def generate_dataloaders(metadata_dict,  start_seed:float, split_ratio:float=0.7):
    """Generate a set of dataloaders by metadta paths, tags, start seed and split ratios. 

    Args:
        metadata_paths (list): Path to list of experiments that need to run. 
        tags (list): List of tags associated with each metadata file. 
        start_seed (float): Start seed to ensure replicability.
        split_ratio (float, optional): Train-test-split ratio. Defaults to 0.7.

    Returns:
        dict: Dictionary containing dataloaders. 
    """
    # Generate seeds. 
    generator = torch.Generator().manual_seed(start_seed)

    # Initialize dataset dictionary
    dataset_dict = {}
    dataset_dict['train_test_match'] = list()

    for tag, metadata_path in metadata_dict.items():
        # Generate dataset
        # print("Tag", tag)
        current_dataset     = CustomDataset(metadata_path = metadata_path)
        outcome_col = tag.split("__")[-1]
        # outcome_col = "_".join(outcome_col[1:])

        # Generate train-test-split ratio. 
        len_current_dataset = len(current_dataset)
        len_trainset        = int(len_current_dataset * split_ratio)
        len_testset         = len_current_dataset - len_trainset
        trainset, testset = random_split(current_dataset, [len_trainset, len_testset], generator=generator)

        # Generate dataset_dict
        dataset_dict[f'{tag}__train']   =  trainset
        dataset_dict[f'{tag}__test']    =  testset
        dataset_dict['train_test_match'].append([f'{tag}__train',f'{tag}__test'])
        if('joint' not in tag):
            dataset_dict['train_test_match'].append([f'joint__{outcome_col}__train',f'{tag}__test'])
    
    return dataset_dict
    
def custom_helper(
    dataset_path,
    root_dir,
    start_seed,
    outcome_cols,
    demographics_col,
    exclude_cols,
    split_ratio):
    
    # Initialize and extract variables. 
    dataset_df = pd.read_csv(dataset_path, sep="\t")
    start_seed = int(start_seed)
    outcome_cols = outcome_cols.split(",")
    exclude_cols = exclude_cols.split(",")

    # Iterate over each demographic and outcome. 
    unique_demographics = list(dataset_df[demographics_col].unique())
    
    # Iterate over the demographic and outcome columns.  
    metadata_dict = {}
    for outcome_col in outcome_cols:
        for demographic in unique_demographics:

            # Iterate over a list of unique outcomes for the current outcome column. 
            dataset_label = f'{demographic}__{outcome_col}'
            metadata_dict[dataset_label] = generate_metadata_file(
                root_dir=root_dir,
                dataset_df = dataset_df, 
                outcome_col = outcome_col,
                outcome_cols = outcome_cols,
                demographic_col = demographics_col,
                demographic = demographic,
                exclude_cols = exclude_cols,
                dataset_label = dataset_label
            )
        # Generate Joint dataset. 
        metadata_dict[f'joint__{outcome_col}'] = generate_metadata_file(
            root_dir=root_dir,
            dataset_df = dataset_df, 
            outcome_col = outcome_col,
            outcome_cols = outcome_cols,
            demographic_col = demographics_col,
            demographic = None,
            exclude_cols = exclude_cols,
            dataset_label = dataset_label
        )
            
    return generate_dataloaders(metadata_dict=metadata_dict, start_seed=start_seed, split_ratio=split_ratio)