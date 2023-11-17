import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import label_binarize
# from aequitas.group import Group
# from aequitas.bias import Bias
# from aequitas.fairness import Fairness
# from aequitas.plotting import Plot


from cnnMCSE.utils.zoo import transfer_helper

def get_AUC(model, loader=None, dataset=None, num_workers:int=0, num_classes:int=10, zoo_model:str=None):
    """Get Area-Under-Curve Metric on a test dataset. 

    Args:
        model (_type_): Model to validate. 
        dataset (_type_): Dataset to use. 

    Returns:
        float: AUC metric. 
    """

    # Using device. 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(f"Using device {device}")

    # Using PreTrained Model. 
    if(zoo_model):
        pretrained_model = transfer_helper(zoo_model)
        pretrained_model = nn.DataParallel(pretrained_model)
        pretrained_model = pretrained_model.to(device=device) 
    else:
        pretrained_model = None

    current_model = model
    current_model = nn.DataParallel(current_model)
    current_model.to(device)
    current_model.eval()

    model_predictions = []
    model_labels      = []

    #print("Generating predictions")
    #print(len(loader))
    #with torch.no_grad():
    #print("Broken here....")
    for index, data in enumerate(loader):
        #print("Running index", index)
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        if(pretrained_model):
            images = pretrained_model(images)

        # images, labels = images.to(DEVICE), labels.to(DEVICE)
        output = current_model(images)
        _, predicted = torch.max(output.data, 1)
        model_predictions = model_predictions + predicted.tolist()
        model_labels = model_labels + labels.tolist()
            # model_predictions.append(predicted.tolist())
            #print('Predictions', predicted.shape)
        # for prediction in predicted:
        #     #print('Predicted', prediction.shape)
        #     model_predictions.append(prediction.tolist())
        # for label in labels:
        #     #print('Labels', label.shape)
        #     model_labels.append(label.tolist())


    #print('Model Labels', len(model_labels))
    #print('Model Predictions',len(model_predictions))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    #print("Calculating AUC.")
    class_list = [i for i in range(num_classes)]
    labels_binarized = label_binarize(model_labels, classes=class_list)
    predictions_binarized = label_binarize(model_predictions, classes=class_list)

    #print("Getting ROC curve. ")
    fpr['micro'], tpr['micro'], _ = metrics.roc_curve(labels_binarized.ravel(), predictions_binarized.ravel())
    roc_auc['micro'] = metrics.auc(fpr['micro'], tpr['micro'])
    #print(roc_auc['micro'])
    return float(roc_auc['micro'])


def get_sAUC(model, loader=None, dataset=None, num_workers:int=0, num_classes:int=10, zoo_model:str=None):
    """Get Area-Under-Curve Metric on a test dataset. 

    Args:
        model (_type_): Model to validate. 
        dataset (_type_): Dataset to use. 

    Returns:
        float: AUC metric. 
    """

    # Using device. 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(f"Using device {device}")

    # Using PreTrained Model. 
    if(zoo_model):
        pretrained_model = transfer_helper(zoo_model)
        pretrained_model = nn.DataParallel(pretrained_model)
        pretrained_model = pretrained_model.to(device=device) 
    else:
        pretrained_model = None

    current_model = model
    current_model = nn.DataParallel(current_model)
    current_model.to(device)
    current_model.eval()

    model_predictions = []
    model_labels      = []

    #print("Generating predictions")
    #print(len(loader))
    #with torch.no_grad():
    #print("Broken here....")
    for index, data in enumerate(loader):
        #print("Running index", index)
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        if(pretrained_model):
            images = pretrained_model(images)

        # images, labels = images.to(DEVICE), labels.to(DEVICE)
        output = current_model(images)
        _, predicted = torch.max(output.data, 1)
        model_predictions = model_predictions + predicted.tolist()
        model_labels = model_labels + labels.tolist()
            # model_predictions.append(predicted.tolist())
            #print('Predictions', predicted.shape)
        # for prediction in predicted:
        #     #print('Predicted', prediction.shape)
        #     model_predictions.append(prediction.tolist())
        # for label in labels:
        #     #print('Labels', label.shape)
        #     model_labels.append(label.tolist())

    roc_dicts = list()
    roc_dict = {}
    unique_labels = list(set(model_labels))

    # print('Unique labels', unique_labels)
    # print('Model Labels', len(model_labels))
    # print('Model Predictions',len(model_predictions))
    # print('Prediction probabilities', )

    roc_dfs = list()
    roc_dict = {}
    for unique_label in unique_labels:

        # print("Running unique label", unique_label)
        current_label_indices = [i for i in range(len(model_labels)) if (model_labels[i] == unique_label)]
        current_labels = [model_label for model_label in model_labels if (model_label == unique_label)]
        current_label_predictions = [model_predictions[i] for i in current_label_indices]
        # print('Current Label Indices', current_label_indices)
        # print('Current Labels', current_labels)
        # print('Current Label Predictions', current_label_predictions)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        #print("Calculating AUC.")
        class_list = [i for i in range(num_classes)]
        labels_binarized = label_binarize(current_label_indices, classes=class_list)
        predictions_binarized = label_binarize(current_label_predictions, classes=class_list)

        #print("Getting ROC curve. ")
        fpr['micro'], tpr['micro'], _ = metrics.roc_curve(labels_binarized.ravel(), predictions_binarized.ravel())
        roc_auc['micro'] = metrics.auc(fpr['micro'], tpr['micro'])
        
        #print('Current AUC', roc_auc['micro'])
        roc_dict['label'] = [unique_label]
        roc_dict[f's_estimands'] = [float(roc_auc['micro'])]
        roc_df = pd.DataFrame(roc_dict)
        roc_dfs.append(roc_df)
    
    roc_df = pd.concat(roc_dfs)
    #print(roc_auc['micro'])
    return roc_df


def get_sAUC2(model, loader=None, dataset=None, num_workers:int=0, num_classes:int=10, zoo_model:str=None):
    """Get Area-Under-Curve Metric on a test dataset. 

    Args:
        model (_type_): Model to validate. 
        dataset (_type_): Dataset to use. 

    Returns:
        float: AUC metric. 
    """

    # Using device. 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(f"Using device {device}")

    # Using PreTrained Model. 
    if(zoo_model):
        pretrained_model = transfer_helper(zoo_model)
        #pretrained_model = nn.DataParallel(pretrained_model)
        pretrained_model = pretrained_model.to(device=device) 
    else:
        pretrained_model = None

    current_model = model
    #current_model = nn.DataParallel(current_model)
    current_model.to(device)
    current_model.eval()

    model_predictions = []
    model_labels      = []

    #print("Generating predictions")
    #print(len(loader))
    #with torch.no_grad():
    #print("Broken here....")
    unique_labels = [i for i in range(num_classes)]
    probability_dict = {}
    probability_tensor = torch.Tensor()
    probability_tensor = probability_tensor.to(device)

    # for unique_label in unique_labels:
    #     probability_dict[unique_label] = list()

    for index, data in enumerate(loader):
        #print("Running index", index)
        print(data)
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        print("Running index", index)
        if(pretrained_model):
            encodings = pretrained_model(images)

        # images, labels = images.to(DEVICE), labels.to(DEVICE)
        #print('Current Model', current_model)
        probs = current_model.module.predict(encodings)
        # probs = torch.sigmoid(output)
        probability_tensor = torch.cat((probability_tensor, probs), dim=0)
        model_labels = model_labels + labels.tolist()

            
            # _, predicted = torch.max(output.data, 1)
            # model_predictions = model_predictions + predicted.tolist()
            # model_labels = model_labels + labels.tolist()
                # model_predictions.append(predicted.tolist())
                #print('Predictions', predicted.shape)
            # for prediction in predicted:
            #     #print('Predicted', prediction.shape)
            #     model_predictions.append(prediction.tolist())
            # for label in labels:
            #     #print('Labels', label.shape)
            #     model_labels.append(label.tolist())

    roc_dicts = list()
    roc_dict = {}
    unique_labels = list(set(model_labels))

    #print('Probability tensor shape', probability_tensor.shape)

    #print('Unique labels', unique_labels)
    #print('Model Labels', len(model_labels))
    #print('Model Predictions',len(model_predictions))
    #print('Prediction probabilities', )
    model_labels_torch = torch.Tensor(model_labels).reshape(probability_tensor.shape[0], 1)

    #print('Model labels shape', model_labels_torch.shape)
    preds_df = torch.cat(((model_labels_torch.to(device)), probability_tensor.to(device)), dim=1)
    #print('Preds df shape', preds_df.shape)

    roc_dfs = list()
    roc_dict = {}
    for unique_label in unique_labels:
        label_probabilities = probability_tensor.select(dim=-1, index=unique_label)
        label_probabilities = label_probabilities.tolist()

        # Source of error is here
        #print("Running unique label", unique_label)
        #current_label_indices = [i for i in range(len(model_labels)) if (model_labels[i] == unique_label)]
        #current_labels = [model_label for model_label in model_labels if (model_label == unique_label)]
        #current_label_predictions = [model_predictions[i] for i in current_label_indices]
        #print('Current Label Indices', current_label_indices)
        #print('Current Labels', current_labels)
        #print('Current Label Predictions', current_label_predictions)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
       # print("Calculating AUC.")
        #class_list = [i for i in range(num_classes)]
        #labels_binarized = label_binarize(current_label_indices, classes=class_list)
        #predictions_binarized = label_binarize(current_label_predictions, classes=class_list)

        #print("Getting ROC curve. ")
        fpr['micro'], tpr['micro'], _ = metrics.roc_curve(model_labels, label_probabilities, pos_label=unique_label)
        roc_auc['micro'] = metrics.auc(fpr['micro'], tpr['micro'])
        
        #print('Current AUC', roc_auc['micro'])
        roc_dict['label'] = [unique_label]
        roc_dict[f's_estimands'] = [float(roc_auc['micro'])]
        roc_df = pd.DataFrame(roc_dict)
        roc_dfs.append(roc_df)
    
    preds_df = pd.DataFrame(preds_df.cpu().detach().numpy())
    roc_df = pd.concat(roc_dfs)
    #print(roc_auc['micro'])
    return roc_df, preds_df 





def get_aucs(models:list, dataset, num_workers:int=0, zoo_model:str=None):
    """Get AUCs for a list of models. 

    Args:
        models (list): _description_
        dataset (_type_): _description_
        num_workers (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    loader  = torch.utils.data.DataLoader(dataset,
                                            batch_size=256,
                                            shuffle=False,
                                            num_workers=num_workers)
    aucs = list()
    for index, model in enumerate(models):
        #print(f"Running model... {index} ")
        auc = get_AUC(model=model, loader=loader, zoo_model=zoo_model)
        aucs.append(auc)
    
    return aucs

def get_sAUCs(models:list, dataset, out_prediction_path:str, num_workers:int=0, zoo_model:str=None):
    """Get AUCs for a list of models. 

    Args:
        models (list): _description_
        dataset (_type_): _description_
        num_workers (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    loader  = torch.utils.data.DataLoader(dataset,
                                            batch_size=256,
                                            shuffle=False,
                                            num_workers=num_workers)
    sauc_dfs = list()
    preds_dfs = list()
    for index, model in enumerate(models):
        #print(f"Running model... {index} ")
        #sauc_df = get_sAUC(model=model, loader=loader, zoo_model=zoo_model)
        sauc_df, preds_df = get_sAUC2(model=model, loader=loader, zoo_model=zoo_model)
        sauc_df['estimands'] = get_AUC(model=model, loader=loader, zoo_model=zoo_model)
        
        preds_df['bootstrap'] = index
        sauc_df['bootstrap'] = index
        
        sauc_dfs.append(sauc_df)
        preds_dfs.append(preds_df)
    
    sauc_df = pd.concat(sauc_dfs)
    preds_dfs = pd.concat(preds_dfs)
    # preds_dfs.to_csv(out_prediction_path, sep="\t", index=False)
    
    return sauc_df, preds_df

def get_frequency(loader):

    all_labels = list()
    for _, data in enumerate(loader):
        _, labels = data
        all_labels = all_labels + labels.tolist()

    # print(all_labels)
    unique_labels = list(set(all_labels))

    label_dfs = list()
    for unique_label in unique_labels:
        label_dict = {}
        num_labels = sum([(label == unique_label) for label in all_labels])
        label_dict['label'] = [unique_label]
        label_dict['frequency'] = [num_labels / len(all_labels)]
        label_df = pd.DataFrame(label_dict)
        label_dfs.append(label_df)
    
    label_df = pd.concat(label_dfs)
    return label_df
    


def get_frequencies(datasets, num_workers:int=0, zoo_model:str=None):
    """Method to get frequencies from a given list of datasets. 

    Args:
        datasets (_type_): List of datasets. 
        num_workers (int, optional): _description_. Defaults to 0.
        zoo_model (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    frq_dfs = list()
    for index, dataset in enumerate(datasets):
        loader  = torch.utils.data.DataLoader(dataset,
                                            batch_size=256,
                                            shuffle=False,
                                            num_workers=num_workers)
        # print(f"Running model... {index} ")
        frq_df = get_frequency(loader=loader)
        frq_df['bootstrap'] = index
        frq_dfs.append(frq_df)
    frq_df = pd.concat(frq_dfs)
    
    return frq_df

def get_sloss(labels:list, s_loss):
    # print('Labels', labels)
    s_loss_dict = {}
    with torch.no_grad():
        sample_mses = torch.mean(s_loss, dim=[i+1 for i in range(len(s_loss.shape)-1)])
        s_loss_dict = {}
        for label, sample_mse in zip(labels, sample_mses):
            #print("Label", label)
            #print("Labels", labels)
            #print("sample-mse", sample_mse)
            #print("Sample_mses", sample_mses)
            #print('Type', type(label))
            #
            if (isinstance(label, list) and len(label) == 1):
                #print("Length", len(label))
                label = label[0]
                # print("Unpacked", label)
                #print("Sample MSE", sample_mse)
            if(label in s_loss_dict):
                s_loss_dict[label].append(sample_mse.item())
            else:
                s_loss_dict[label] = [sample_mse.item()]
        for key, value in s_loss_dict.items():
            s_loss_dict[key] = [np.mean(value)]
    
    return s_loss_dict
    
def metric_helper(models, 
    metric_type:str, 
    datasets=None, 
    dataset=None, 
    loader=None, 
    num_workers:int=1, 
    zoo_model:str=None, 
    labels=None,
    s_loss=None,
    out_prediction_path=None):
    """Select which metric to use. 

    Args:
        models (_type_): Models. 
        metric_type (str): Metric Type. 
        dataset (_type_, optional): Which validation dataset to use. Defaults to None.
        loader (_type_, optional): Which dataloader to use. Defaults to None.
        num_workers (int, optional): Number of workers. Defaults to 1.

    Returns:
        list: List of validated metrics
    """
    if(metric_type == "AUC"):
        return get_aucs(models=models, dataset=dataset, num_workers=num_workers, zoo_model=zoo_model)
    
    if(metric_type == "sAUC"):
        return get_sAUCs(models=models, dataset=dataset, num_workers=num_workers, zoo_model=zoo_model, out_prediction_path=out_prediction_path)
    
    if(metric_type == "frequencies"):
        return get_frequencies(datasets=datasets, num_workers=num_workers, zoo_model=zoo_model)
    
    if(metric_type == "sloss"):
        return get_sloss(labels=labels, s_loss=s_loss)
        #return get_AUC(model=model, dataset=dataset, loader=loader)