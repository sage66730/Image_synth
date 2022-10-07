import torch

from model import *
from dataset import *

# punish JawOpen MouthClose MouthFunnel MouthPucker
def WeightedMSE_4(pred, label):
    a = 1
    return a*(torch.sum((pred[17:21] - label[17:21]) ** 2)) + (1-a)*(torch.sum((pred[:17] - label[:17]) ** 2)+torch.sum((pred[21:61] - label[21:61]) ** 2))

def WeightedMSE_MJ(pred, label):
    a = 1
    return a*(torch.sum((pred[14:41] - label[14:41]) ** 2)) + (1-a)*(torch.sum((pred[:14] - label[:14]) ** 2)+torch.sum((pred[41:61] - label[41:61]) ** 2))

def WeightedMSE_M(pred, label):
    a = 0.8
    return a*(torch.sum((pred[18:41] - label[18:41]) ** 2)) + (1-a)*(torch.sum((pred[:18] - label[:18]) ** 2)+torch.sum((pred[41:61] - label[41:61]) ** 2))

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
    if args.loss_fn == "WeightedMSE_4": return WeightedMSE_4
    if args.loss_fn == "WeightedMSE_M": return WeightedMSE_M
    if args.loss_fn == "WeightedMSE_MJ": return WeightedMSE_MJ

    print("loss function name incorrect")
    return None

def get_optimizer(model, args):
    if args.optimizer == "Adam": return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "SGD": return torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.5)

    print("optimizer name incorrect")
    return None

def get_scheduler(optimizer, args):
    if args.scheduler == "LambdaLR": return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1*(0.9**epoch), last_epoch=-1)

    print("scheduler name incorrect")
    return None
