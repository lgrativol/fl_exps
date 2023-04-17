import flwr as fl
from flwr.common.typing import Scalar
import torch
import numpy as np
from pathlib import Path
from typing import Dict
from dataset_utils import get_dataloader,dict_tranforms
from utils import *
import os
from binaryconnect import BC
from args import args

# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model : str, dataset_info, saddr : str ,cid: str, fed_dir_data: str, features_maps : int = 32, only_cpu : bool = False):
        self.saddr = saddr
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.dataset = dataset_info["name"]
        # Instantiate model
        self.net = model(features_maps, dataset_info["input_shape"], dataset_info["num_classes"],batchn=False)
        if(args.bnn):
            self.net = BC(self.net)
        #self.net = Net()

        # Determine device
        if(only_cpu): #brute force
            os.environ["CUDA_VISIBLE_DEVICES"]=""
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")

    def get_parameters(self, config):
        return get_params(self.net)
        
    def fit(self, parameters, config):

        if(args.prune_srv):
            parameters = prune_threshold(parameters)

        set_params(self.net, parameters)

        lr = config["cl_lr"]
        momentum = config["cl_momentum"]
        # Load data for this client and get trainloader
        num_workers = 2
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
            transform = dict_tranforms[self.dataset]
        )

        # Send model to device
        self.net.to(self.device)

        # Train
        train(self.net, trainloader, epochs=config["epochs"], 
        device=self.device,lr=lr,momentum=momentum )

        sparsity = 0
        params = get_params(self.net)

        if(args.prune):
            params = prune_threshold(params)

        if(args.layer_sparsity):
            sparsity = layer_sparsity(params)
            
        return params, len(trainloader.dataset), {"sparsity":sparsity}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = 2
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=50, 
            workers=num_workers,transform = dict_tranforms[self.dataset])

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

    def start_client(self):
        # Start Flower client
        fl.client.start_numpy_client(server_address=self.saddr, client=self)
