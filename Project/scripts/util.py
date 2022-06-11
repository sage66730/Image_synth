import torch

from model import ObjModel1
from dataset import ObjDataset

def get_model(args):
    if args.model == "ObjModel1": return ObjModel1(5023*3,61)

    print("model name incorrect")
    return None

def get_dataset(args):
    if args.dataset == "ObjDataset": return ObjDataset(args.dataset_path, args.target)

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