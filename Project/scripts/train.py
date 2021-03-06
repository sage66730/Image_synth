import os
import pytz
import torch
import argparse
from tqdm import tqdm
from statistics import mean
from datetime import datetime
from torch.utils.data import DataLoader

from util import * 

def setup_dir(args):
    timezone = pytz.timezone("America/Los_Angeles")
    start_dt = (datetime.now().astimezone(timezone)).strftime("%Y-%m-%d_%H:%M:%S")
    dir_path = f"{args.save_path}/{start_dt}"
    os.mkdir(dir_path)
    print(f"start traing at {start_dt}")

    with open(f"{dir_path}/info.txt", 'w') as fp:
        fp.write(f"model {args.model}\n")
        fp.write(f"dataset {args.dataset}\n")
        fp.write(f"target {args.target}\n")
        fp.write(f"loss_fn {args.loss_fn}\n")
        fp.write(f"optimizer {args.optimizer}\n")
        fp.write(f"batch_size {args.batch_size}\n")
        fp.write(f"learning_rate {args.learning_rate}\n")
        fp.write(f"message {args.message}\n")
        #fp.write(f" {args.}\n")
    return dir_path

def main(args):
    args.target = read_training_set(args.dataset_path) 
    dir_path = setup_dir(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # initialize
    model = get_model(args).to(device)
    dataset = get_dataset(args)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loss_fn = get_loss_fn(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    # train
    hist = {"loss":[]}
    stop_cnt = 0
    tq = tqdm(range(args.epoch))
    for e in tq:
        #epoch train
        losses = []
        for _, data, label in train_dataloader:
            #data = data.to(device)
            #label = label.to(device)
            pred = model(data)
            loss = loss_fn(pred, label)

            losses.append(loss.to("cpu").item())
            tq.set_postfix({'mean_loss': mean(losses)})

            model.zero_grad()
            loss.backward()
            optimizer.step()
        #scheduler.step()

        epo_loss = mean(losses)

        #early stopping
        if hist["loss"] and abs(epo_loss - hist["loss"][-1]) < 0.01:
            stop_cnt += 1
            if stop_cnt == 10:
                print(f"early stopping at {e}")
                torch.save(model, f"{dir_path}/{e:03d}_{epo_loss}.pth")
                break
        else: stop_cnt = 0

        #save model
        if e%(args.save_freqency) == 0 or e == args.epoch-1:
            torch.save(model, f"{dir_path}/{e:03d}_{epo_loss}.pth")

        #hist recording
        hist["loss"].append(epo_loss)

    # record
    with open(f"{dir_path}/info.txt", 'a') as fp:
        losses = hist["loss"]
        print(losses)
        fp.write(f"losses {losses}\n")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # env
    parser.add_argument("--dataset_path", type=str, default="/home/sage66730/dataset", help="path to the root dir of dataset")
    parser.add_argument("--save_path", type=str, default="/home/sage66730/Image_synth/Project/models", help="path to the save dir for trianed checkpoint")
    parser.add_argument("--message", type=str, default="No message", help="custom message that would be displayed in info.txt")
    
    # configuration
    parser.add_argument("--model", type=str, default="ObjMJModel1", help="the model to be trained")
    parser.add_argument("--dataset", type=str, default="ObjMJDataset", help="the dataset to be used")
    parser.add_argument("--loss_fn", type=str, default="WeightedMSE_M", help="the loss fn to be used")
    parser.add_argument("--optimizer", type=str, default="Adam", help="the optimizer to be used")
    parser.add_argument("--scheduler", type=str, default="LambdaLR", help="the scheduler to be used")

    # training para
    parser.add_argument("--epoch", type=int, default=20, help="number of epoch to train")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size to train")
    parser.add_argument("--save_freqency", type=int, default=1, help="freqency to save result")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate to train")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay to train")
    
    main(parser.parse_args()) 