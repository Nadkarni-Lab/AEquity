
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from aequity.metrics import metric_helper


def get_estimators(
    model,
    training_data,
    sample_size:int,
    initial_weights:str, 
    batch_size:int=1,
    bootstraps:int=1,
    start_seed:int=42,
    shuffle:bool=False,
    num_workers:int=1,
    zoo_model:str=None,
    frequency:bool=False,
    stratified:bool=False,
    n_epochs:int=1,
    current_bootstrap:int=None,
    sampler_mode:str=None,
    input_size:int=None,
    hidden_size:int=None):
    """Method to get estimators for convergence samples. 

    Args:
        model (_type_): A model. 
        training_data (_type_): Training data. 
        sample_size (int): Sample size to estimate at. 
        batch_size (int, optional): Batch size. Defaults to 4.
        bootstraps (int, optional): Number of bootstraps. Defaults to 1.
        start_seed (int, optional): Seed. Defaults to 42.
        shuffle (bool, optional): Shuffle the dataset. Defaults to False.
        num_workers (int, optional): Number of workers. Defaults to 1.
        frequency (bool, optional): Whether to calculate frequencies or not. 

    Returns:
        list: List of losses. 
    """

    # Determine which device is being used. 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Generate zoo models. 
    pretrained_model = None

    # run across all the bootstraps
    losses = list()
    train_subsets = list()
    s_losses = {}

    for i in range(bootstraps):
        #print("Running loop ", i)

        # Create a generator for replicability. 
        #print("Generating generator")
        if(current_bootstrap):
            generator = torch.Generator().manual_seed(start_seed+current_bootstrap)
        else:
            generator = torch.Generator().manual_seed(start_seed+i)

        # generate a unique training subset.
        #print("Creating training subset")
        train_subset, _ = random_split(training_data, lengths = [sample_size, len(training_data) - sample_size], generator=generator)
        train_subsets.append(train_subset)
        

        # Create a training dataloader. 
       # print("Create a training dataloader. ")
        trainloader = torch.utils.data.DataLoader(train_subset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True)
        if(sampler_mode):
            train_sampler = weighted_sampler(
                subset=train_subset,
                mode=sampler_mode
            )
            trainloader = torch.utils.data.DataLoader(train_subset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              sampler=train_sampler)


        # Initialize current model. 
        if(input_size):
            if(hidden_size == None):
                current_model = model(input_size=input_size)
            else:
                current_model = model(input_size=input_size, hidden_size_one=hidden_size, hidden_size_two=hidden_size, hidden_size_three=hidden_size)
        else:
            current_model= model()
        current_model.load_state_dict(torch.load(initial_weights))

        # Parallelize current model. 
        #print("Parallelize current model. ")
        current_model = nn.DataParallel(current_model)
        current_model.to(device)

        # Set model in training mode. 
        #print("Set model in training mode. ")
        current_model.train()

        # Generate mean-squared error loss criterion
        #print("Generate mean-squared error loss criterion.")
        if(stratified):
            s_criterion = nn.MSELoss(reduction='none')
            
        criterion = nn.MSELoss()
        #else:
        #   

        # Optimize model with stochastic gradient descent. 
        #print("Optimize model with stochastic gradient descent. ")
        optimizer_model = optim.SGD(current_model.parameters(), lr=0.001, momentum=0.9)

        # Set up training loop
        running_loss = 0.0
        s_running_loss = dict()
        
        # iterate over training subset. 
        for i in range(n_epochs):
            for j, data in enumerate(trainloader):
                #print("Testing data ", j)
                #print('Data', data)
                #print("Running batch" , i)

                # Get data
                inputs, labels = data
                #inputs = inputs.flatten()
                inputs, labels = inputs.to(device), labels.to(device)

                if(pretrained_model):
                    inputs = pretrained_model(inputs)

                # Zero parameter gradients
                optimizer_model.zero_grad()

                # Forward + backward + optimize
                # print(inputs.shape)
                outputs = current_model(inputs)
                # print('Output shape', outputs.shape)
                # Accomodate for intra-model flattening. 
                inputs = inputs.reshape(outputs.shape)
                # print('Input shape', inputs.shape)
                loss = criterion(outputs, inputs)
                #print("Inputs", inputs)
                #print("Outputs", outputs)
                #print("Loss", loss)

                if(stratified and (i == n_epochs - 1)):
                    #print("Running stratified loss...")
                    s_loss = s_criterion(outputs, inputs)
                    labels = labels.tolist()
                    s_loss_dict = metric_helper(metric_type="sloss", s_loss=s_loss, labels=labels, models=None)

                loss.backward()
                optimizer_model.step()

                running_loss += loss.item()

                #print("Adding losses to running losses...")
                if(i == n_epochs - 1):
                    for key, value in s_loss_dict.items():
                        if(key in s_running_loss):
                            s_running_loss[key] += s_loss_dict[key][0]
                        else:
                            s_running_loss[key] = s_loss_dict[key][0]


        # Add loss
        loss = running_loss / sample_size
        losses.append(float(loss))

        if(stratified):
            for key, value in s_running_loss.items():
                s_loss = value / sample_size
                if(key in s_losses):
                    s_losses[key].append(s_loss)
                else:
                    s_losses[key] = [s_loss]
            
  


    if(frequency == True):
        loss_dict = {
            'estimators': losses
        }
        loss_df = pd.DataFrame(loss_dict)
        loss_df = loss_df.reset_index()
        loss_df['bootstrap'] = loss_df['index']
        frequency_df = metric_helper(models=None, metric_type="frequencies", datasets=train_subsets, num_workers=0)
        #print(frequency_df)
        #print(loss_df)
        merged_df = frequency_df.merge(loss_df, on='bootstrap')
        #print(merged_df)
        #merged_df = pd.concat([loss_df, frequency_df], axis=1, ignore_index=True)
        
        if(stratified):
            # print(s_losses)
            for key, value in s_losses.items():
                while(len(s_losses[key]) < bootstraps):
                    s_losses[key].append(None)
            
            val_dict = {}
            val_dict['bootstrap'] = []
            val_dict['label'] = []
            val_dict['s_estimators'] = []

            for key, item in s_losses.items():
                val_dict['bootstrap'] += ([i for i in range(len(item))])
                val_dict['label'] += ([key for i in range(len(item))])
                val_dict['s_estimators'] += (item)

            s_mcse_df = pd.DataFrame.from_dict(val_dict)
            #s_mcse_df = s_mcse_df.reset_index()
            #s_mcse_df['label']  = s_mcse_df['index']
            #print(s_mcse_df)
            #print(merged_df)
            merged_df = merged_df.merge(s_mcse_df, on=['label', 'bootstrap'])

        merged_df['sample_size'] = sample_size
        return merged_df
    
    else:
        return losses

    gc.collect()
    return merged_df


def get_estimands(
    model,
    training_data,
    validation_data,
    sample_size,
    initial_weights:str,
    out_prediction_path:str,
    batch_size:int=1,
    bootstraps:int=1,
    start_seed:int=42,
    shuffle:bool=False,
    metric_type:str="AUC",
    num_workers:int=1,
    zoo_model:str=None,
    n_epochs:int=1,
    current_bootstrap:int=None,
    sampler_mode:str=None,
    input_size:int=None,
    hidden_size:int=None,
    output_size:int=None
    ):
    """Method to generate estimands. 

    Args:
        model (nn.Module): Model to train data on. 
        training_data (Dataset): Training data. 
        validation_data (Dataset): Validation data. 
        sample_size (int): Sample size. 
        initial_weights (str): Initial weights. 
        batch_size (int, optional): Batch size. Defaults to 4.
        bootstraps (int, optional): Number of bootstraps. Defaults to 1.
        start_seed (int, optional): Start seed. Defaults to 42.
        shuffle (bool, optional): Shuffle. Defaults to False.
        metric_type (str, optional): Metric type. Defaults to "AUC".

    Returns:
        list: List of estimation. 
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    if(zoo_model):
        pretrained_model = transfer_helper(zoo_model)
        pretrained_model = pretrained_model.to(device=device) 
    else:
        pretrained_model = None

    # print("Getting estimands")
    models = list()
    model_paths = list()
    metrics = list()

    for i in range(bootstraps):
        #print("Running estimands ", i)
        # Create a generator for replicability. 
        #print("Create a generator for replicability.")
        if(current_bootstrap):
            generator = torch.Generator().manual_seed(start_seed+current_bootstrap)
        else:
            generator = torch.Generator().manual_seed(start_seed+i)

        # generate a unique training subset.
        #print("generate a unique training subset..")
        train_subset, _ = random_split(training_data, lengths = [sample_size, len(training_data) - sample_size], generator=generator)
        
        # Create a training dataloader. 
        #print("Create a training dataloader")
        trainloader = torch.utils.data.DataLoader(train_subset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True)
        if(sampler_mode):
            train_sampler = weighted_sampler(
                subset=train_subset,
                mode=sampler_mode
            )
            trainloader = torch.utils.data.DataLoader(train_subset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              sampler=train_sampler)


        # Initialize current model. 
        #print("Initialize current model. ")
        if(zoo_model == "alexnet"): 
            current_model = model(input_size=9216)
        elif(zoo_model == "radimagenet"):
            current_model = Classifier(num_class=10)
        else:
            if(input_size):
                if(hidden_size == None):
                    current_model = model(input_size=input_size)
                elif(hidden_size != None and output_size == None):
                    current_model = model(input_size=input_size, hidden_size_one=hidden_size, hidden_size_two=hidden_size, hidden_size_three=hidden_size)
                elif(hidden_size != None and output_size != None):
                    current_model = model(input_size=input_size, hidden_size_one=hidden_size, hidden_size_two=hidden_size, hidden_size_three=hidden_size, output_size=output_size)
            else:
                current_model = model()
        current_model.load_state_dict(torch.load(initial_weights))

        # Parallelize current model. 
        #print("Parallelize current model.")
        current_model = nn.DataParallel(current_model)
        current_model.to(device)

        # Set model in training mode. 
        #print("Set model in training mode")
        current_model.train()

        # Assign CrossEntropyLoss and stochastic gradient descent optimizer. 
        #print("Run cross entropy loss")
        criterion = nn.CrossEntropyLoss()
        optimizer_model = optim.SGD(current_model.parameters(), lr=0.001, momentum=0.9)

        # Train the model.  
        #print("Running loop for estimands")
        for epoch in range(n_epochs):
            #print("Running epoch... ", epoch)
            for j, data in enumerate(trainloader):
                #print("Testing data ", j)
                # Get data
                inputs, labels = data
                #inputs = torch.flatten(inputs, start_dim=1)
                inputs, labels = inputs.to(device), labels.to(device)

                if(pretrained_model):
                    inputs = pretrained_model(inputs)


                # Zero parameter gradients
                optimizer_model.zero_grad()

                # Forward + backward + optimize
                outputs = current_model(inputs)

                #print(outputs.shape)
                #print(labels.shape)
                #print(labels)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_model.step()
        
        models.append(current_model)

    #print("Evaluating models... ")
    #print(len(models))
    metrics, preds_df = metric_helper(
        models = models,
        dataset=validation_data,
        metric_type=metric_type,
        num_workers=num_workers,
        zoo_model=zoo_model, 
        out_prediction_path=out_prediction_path
    )
    gc.collect()
    return metrics, preds_df