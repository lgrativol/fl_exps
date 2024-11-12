import argparse

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

## FL
## FL
parser.add_argument("--num_rounds", type=int, default=2,help="number of rounds for a federated learning training")
parser.add_argument("--num_clients", type=int, default=10,help="number of clients")
parser.add_argument("--alpha",type=float, default=1,help="alpha used for LDA")
parser.add_argument("--alpha_inf",action="store_true",help="special flag to use alpha as infinity") # uniform
parser.add_argument("--val_ratio", type=float, default=0.0,help="validationd dataset split")
parser.add_argument("--dataset", type = str, default='cifar10',help="which dataset to use")
parser.add_argument("--samp_rate", type=float, default=0.2,help="client's sample rate")
parser.add_argument("--skip_gen_training", action="store_true",help="to skip training pt data")

## Model
parser.add_argument("--model", type = str, default='resnet8',help="model to use (resnet18, resnet20,qresnet12)")
parser.add_argument("--feature_maps", type=int, default=16,help="number of feature maps for the model")
parser.add_argument("--wbits", type=int,default=0,help="number of bits for QAT")
parser.add_argument("--bnn", action="store_true",help="convert model to binary")

## Client
parser.add_argument("--cl_lr", type=float, default=0.01,help="client's learning rate")
parser.add_argument("--cl_momentum", type=float, default=0.9,help="client's momentum")
parser.add_argument("--cl_epochs", type=int, default=1,help="number of local epochs in each client")
parser.add_argument("--cl_bs", type=int, default=16,help="client's batch size")
parser.add_argument("--only_cpu", action="store_true",help="to force the use of only cpu in the client")

## Prune
parser.add_argument("--prune", action="store_true",help="flag to activate pruning")
parser.add_argument("--prune_srv", action="store_true",help="flag to activate pruning on the server side")
parser.add_argument("--prate", type=float, default=0.1,help="pruning rate")
parser.add_argument("--prate_min", type=float, default=0.0,help="minimum prate fpr the layer pruning scheme")
parser.add_argument("--layer_sparsity",action="store_true",help="sparsity per layer")

# parse input arguments
args = parser.parse_args()
