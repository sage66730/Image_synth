import os
import csv
import glob
import torch
import numpy
import shutil
import argparse
from tqdm import tqdm
from statistics import mean
from datetime import datetime
from torch.utils.data import DataLoader

from util import * 

def setup_dir(args):
    start_dt = (args.model_path).split("/")[-1]
    dir_path = f"{args.save_path}/{start_dt}"
    if os.path.isdir(dir_path):
        print("over writing") 
        shutil.rmtree(dir_path) 
    os.mkdir(dir_path)

    return dir_path

def get_arguments(args):
    configs = dict()
    with open(f"{args.model_path}/info.txt") as fp:
        for line in fp:
            line = line.strip().split(" ")
            configs[line[0]] = line[1]
    
    args.model = configs["model"]
    args.dataset = configs["dataset"]
    args.loss_fn = configs["loss_fn"]

    return args

def get_best_model(args):
    models = glob.glob(f"{args.model_path}/*.pth")
    models = [(float(model.split("/")[-1][4:-4]),model) for model in models]
    models = sorted(models, key=lambda x: x[0])

    return models[0][1]

def save_reult(path, preds):
    for subject in preds.keys():
        os.mkdir(f"{path}/{subject}")
        for sentence in preds[subject].keys():
            os.mkdir(f"{path}/{subject}/{sentence:02d}")
            csvfile = open(f"{path}/{subject}/{sentence:02d}/output.csv", "w", newline='')
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Timecode","BlendShapeCount",
                                "EyeBlinkLeft",	"EyeLookDownLeft",	"EyeLookInLeft",	"EyeLookOutLeft",	"EyeLookUpLeft", "EyeSquintLeft",	"EyeWideLeft",	"EyeBlinkRight",	
                                "EyeLookDownRight",	"EyeLookInRight",	"EyeLookOutRight",	"EyeLookUpRight",	"EyeSquintRight",	"EyeWideRight",	"JawForward",	"JawRight",	
                                "JawLeft",	"JawOpen",	"MouthClose",	"MouthFunnel",	"MouthPucker",	"MouthRight",	"MouthLeft",	"MouthSmileLeft",	
                                "MouthSmileRight",	"MouthFrownLeft",	"MouthFrownRight",	"MouthDimpleLeft",	"MouthDimpleRight",	"MouthStretchLeft",	"MouthStretchRight", "MouthRollLower",	
                                "MouthRollUpper",	"MouthShrugLower",	"MouthShrugUpper",	"MouthPressLeft",	"MouthPressRight",	"MouthLowerDownLeft",	"MouthLowerDownRight",	"MouthUpperUpLeft",	
                                "MouthUpperUpRight",	"BrowDownLeft",	"BrowDownRight",	"BrowInnerUp",	"BrowOuterUpLeft",	"BrowOuterUpRight",	"CheekPuff",	"CheekSquintLeft",	
                                "CheekSquintRight",	"NoseSneerLeft",	"NoseSneerRight",	"TongueOut",	"HeadYaw",	"HeadPitch",	"HeadRoll",	"LeftEyeYaw",	
                                "LeftEyePitch",	"LeftEyeRoll",	"RightEyeYaw",	"RightEyePitch",	"RightEyeRoll"])
            for r, row in enumerate(preds[subject][sentence]):
                m, s = divmod(r,60)
                new_row = [f"00:00:{m:02d}:{s:02d}.000","61"] + row
                csvwriter.writerow(new_row)
            
    return

def main(args):
    args.target = read_testing_set(args.dataset_path)
    dir_path = setup_dir(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # initialize 
    args = get_arguments(args)
    model = get_model(args).to(device)
    dataset = get_dataset(args)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = get_loss_fn(args)

    best = get_best_model(args)
    model = torch.load(best)
    model.eval()
    print(f"Using model {best}")
    print(f"Target: {args.target}")
    
    # test
    preds = { target:{i:[] for i in range(40)} for target in args.target}
    losses = []
    tq = tqdm(test_dataloader)
    for info, data, label in tq:
        with torch.no_grad():
            #data = data.to(device)
            #label = label.to(device)
            pred = model(data)
            loss = loss_fn(pred, label)

        subject = info[0][0]
        sentence = int(info[1][0])

        pred = pred.squeeze()
        preds[subject][sentence].append(pred.to("cpu").numpy().tolist())
        losses.append(loss.to("cpu").item())

    mean_loss = mean(losses)

    #save result
    save_reult(dir_path, preds)

    # print loss 
    print(mean_loss)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # env
    parser.add_argument("--dataset_path", type=str, default="/home/sage66730/dataset", help="path to the root dir of dataset")
    parser.add_argument("--model_path", type=str, default="/home/sage66730/Project/models/10-06-2022_13:07:30", help="path to the saved model dir for testing")
    parser.add_argument("--save_path", type=str, default="/home/sage66730/Image_synth/Project/results", help="path to the save dir for testing result")
    
    # configuration
    parser.add_argument("--model", type=str, default="ObjModel1", help="the model to be trained")
    parser.add_argument("--dataset", type=str, default="ObjDataset", help="the dataset to be used")

    main(parser.parse_args()) 