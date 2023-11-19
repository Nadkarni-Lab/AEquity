import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

class A3(nn.Module):
    """Fully connected network autoencoder. 
    """
    def __init__(self, input_size: int=784, 
        hidden_size_one: int = 1024, 
        hidden_size_two : int = 512, 
        hidden_size_three: int = 256, 
        latent_classes: int = 2):
        super(A3, self).__init__()
        self.input_size = input_size
        self.hidden_size_one = hidden_size_one
        self.hidden_size_two = hidden_size_two
        self.hidden_size_three = hidden_size_three

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size_one),
            nn.ReLU(True), 
            nn.Linear(hidden_size_one, hidden_size_two), 
            nn.ReLU(True), 
            nn.Linear(hidden_size_two, hidden_size_three), 
            nn.ReLU(True), 
            nn.Linear(hidden_size_three, latent_classes))

        self.decoder = nn.Sequential(
            nn.Linear(latent_classes, hidden_size_three),
            nn.ReLU(True),
            nn.Linear(hidden_size_three, hidden_size_two),
            nn.ReLU(True),
            nn.Linear(hidden_size_two, hidden_size_one),
            nn.ReLU(True),
            nn.Linear(hidden_size_one, input_size))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_2(self, x):
        x = self.encoder(x)
        return x 

class FCN(nn.Module):
    """Fully connected neural network classifier. 
    """
    def __init__(self, input_size: int=784, 
        hidden_size_one: int = 1024, 
        hidden_size_two : int = 512, 
        hidden_size_three: int = 256, 
        output_size: int = 10):
        super(FCN, self).__init__()
        self.input_size = input_size
        self.hidden_size_one = hidden_size_one
        self.hidden_size_two = hidden_size_two
        self.hidden_size_three = hidden_size_three
        self.output_size = output_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size_one),
            nn.ReLU(True), 
            nn.Linear(hidden_size_one, hidden_size_two), 
            nn.ReLU(True), 
            nn.Linear(hidden_size_two, hidden_size_three), 
            nn.ReLU(True), 
            nn.Linear(hidden_size_three, output_size))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        x = F.log_softmax(x, dim=1)
        return x

    def forward_2(self, x):
        x = self.encoder(x)
        return x     
    
    def predict(self, x):
        x = torch.flatten(x, start_dim = 1)
        x = self.encoder(x)
        x = F.softmax(x, dim=1)
        return x

def model_helper(model:str, initial_weights_dir:str, input_dim=None, hidden_size=None, output_size=None)->nn.Module:
    """Method to return torch model. 

    Args:
        model (str): Name of the model.

    Returns:
        nn.Module: Model to return. 
    """
    if(model == "A3" and input_dim == None):
        a3 = A3()
        initial_weights_path = os.path.join(initial_weights_dir, model + '.pt')
        torch.save(a3.state_dict(), initial_weights_path)
        return A3, initial_weights_path
    
    elif(model == "FCN" and input_dim == None):
        fcn = FCN()
        initial_weights_path = os.path.join(initial_weights_dir, model + '.pt')
        torch.save(fcn.state_dict(), initial_weights_path)
        return FCN, initial_weights_path
    
    elif(model == "A3" and input_dim != None and hidden_size == None):
        a3 = A3(input_size=input_dim)
        initial_weights_path = os.path.join(initial_weights_dir, model + str(input_dim) + '.pt')
        torch.save(a3.state_dict(), initial_weights_path)
        return A3, initial_weights_path
    
    elif(model == "FCN" and input_dim  !=None and hidden_size == None):
        fcn = FCN(input_size=input_dim)
        initial_weights_path = os.path.join(initial_weights_dir, model + str(input_dim) + '.pt')
        torch.save(fcn.state_dict(), initial_weights_path)
        return FCN, initial_weights_path
    

    elif(model == "A3" and input_dim != None and hidden_size != None):
        a3 = A3(input_size=input_dim, 
        hidden_size_one=hidden_size, 
        hidden_size_two = hidden_size, 
        hidden_size_three= hidden_size)
        a3.apply(init_xavier)
        initial_weights_path = os.path.join(initial_weights_dir, model + str(input_dim) + str(hidden_size) + '.pt')
        torch.save(a3.state_dict(), initial_weights_path)
        return A3, initial_weights_path
    
    elif(model == "FCN" and input_dim  !=None and hidden_size != None and output_size == None):
        fcn = FCN(input_size=input_dim, hidden_size_one=hidden_size, 
        hidden_size_two = hidden_size, 
        hidden_size_three= hidden_size)
        fcn.apply(init_xavier)
        initial_weights_path = os.path.join(initial_weights_dir, model  + str(input_dim) + str(hidden_size) +  '.pt')
        torch.save(fcn.state_dict(), initial_weights_path)
        return FCN, initial_weights_path
    

    elif(model == "A3" and input_dim != None and hidden_size != None and output_size != None):
        a3 = A3(input_size=input_dim, 
        hidden_size_one=hidden_size, 
        hidden_size_two = hidden_size, 
        hidden_size_three= hidden_size)
        a3.apply(init_xavier)
        initial_weights_path = os.path.join(initial_weights_dir, model + str(input_dim) + str(hidden_size) + '.pt')
        torch.save(a3.state_dict(), initial_weights_path)
        return A3, initial_weights_path
    
    elif(model == "FCN" and input_dim  !=None and hidden_size != None and output_size != None):
        fcn = FCN(input_size=input_dim, hidden_size_one=hidden_size, 
        hidden_size_two = hidden_size, 
        hidden_size_three= hidden_size,
        output_size=output_size)
        fcn.apply(init_xavier)
        initial_weights_path = os.path.join(initial_weights_dir, model  + str(input_dim) + str(hidden_size) +  '.pt')
        torch.save(fcn.state_dict(), initial_weights_path)
        return FCN, initial_weights_path