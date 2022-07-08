import torch

from model import *
from dataset import *

def read_training_set(dataset_path):
    training_set = []
    with open(f"{dataset_path}/training_set.txt") as fp:
        for line in fp:
            training_set.append(line.strip())
    return training_set

def read_testing_set(dataset_path):
    testing_set = []
    with open(f"{dataset_path}/testing_set.txt") as fp:
        for line in fp:
            testing_set.append(line.strip())
    return testing_set

def get_model(args):
    if args.model == "ObjModel1": return ObjModel1(5023*3,61)
    if args.model == "ObjMJModel1": return ObjMJModel1(5023*3,61)

    print("model name incorrect")
    return None

def get_dataset(args):
    if args.dataset == "ObjDataset": return ObjDataset(args.dataset_path, args.target)
    if args.dataset == "ObjMJDataset": return ObjMJDataset(args.dataset_path, args.target)

    print("dataset name incorrect")
    return None

def get_loss_fn(args):
    if args.loss_fn == "MSE": return torch.nn.MSELoss(reduction='sum')

    print("loss function name incorrect")
    return None

def get_optimizer(model, args):
    if args.optimizer == "Adam": return torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("optimizer name incorrect")
    return None