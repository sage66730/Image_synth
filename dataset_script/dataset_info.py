from ast import arg
import os
import csv
import glob
import argparse
from tqdm import tqdm

blend_names = ["EyeBlinkLeft",	"EyeLookDownLeft",	"EyeLookInLeft",	"EyeLookOutLeft",	"EyeLookUpLeft", "EyeSquintLeft",	"EyeWideLeft",	"EyeBlinkRight",	
                "EyeLookDownRight",	"EyeLookInRight",	"EyeLookOutRight",	"EyeLookUpRight",	"EyeSquintRight",	"EyeWideRight",	"JawForward",	"JawRight",	
                "JawLeft",	"JawOpen",	"MouthClose",	"MouthFunnel",	"MouthPucker",	"MouthRight",	"MouthLeft",	"MouthSmileLeft",	
                "MouthSmileRight",	"MouthFrownLeft",	"MouthFrownRight",	"MouthDimpleLeft",	"MouthDimpleRight",	"MouthStretchLeft",	"MouthStretchRight", "MouthRollLower",	
                "MouthRollUpper",	"MouthShrugLower",	"MouthShrugUpper",	"MouthPressLeft",	"MouthPressRight",	"MouthLowerDownLeft",	"MouthLowerDownRight",	"MouthUpperUpLeft",	
                "MouthUpperUpRight",	"BrowDownLeft",	"BrowDownRight",	"BrowInnerUp",	"BrowOuterUpLeft",	"BrowOuterUpRight",	"CheekPuff",	"CheekSquintLeft",	
                "CheekSquintRight",	"NoseSneerLeft",	"NoseSneerRight",	"TongueOut",	"HeadYaw",	"HeadPitch",	"HeadRoll",	"LeftEyeYaw",	
                "LeftEyePitch",	"LeftEyeRoll",	"RightEyeYaw",	"RightEyePitch",	"RightEyeRoll"]

def blend_averge(args):
    blend_sum = [0 for _ in range(61)]
    blend_count = 0

    subjects = os.listdir(f"{args.dataset_path}/label")
    for subject in tqdm(subjects):
        sentences = os.listdir(f"{args.dataset_path}/label/{subject}")
        for sentence in sentences:
            csv_path = glob.glob(f"{args.dataset_path}/label/{subject}/{sentence}/aligned.csv")
            fp = open(csv_path[0], newline='')
            cdata = list(csv.reader(fp))

            blend_count += len(cdata)-1
            for i, row in enumerate(cdata):
                if i !=0 :
                    for j in range(61): blend_sum[j] += float(row[j+2])

    blend_rank = [ (i, blend_names[i], blend_sum[i]/blend_count) for i in range(61)]
    blend_rank = sorted(blend_rank, key=lambda x:x[2], reverse=True)

    print(f"total blend count: {blend_count}")
    for row in blend_rank:
        print(f"{row[0]:02d}:{row[1]:20}{row[2]}")

    return 

def print_obj(args):
    with open(f"{args.dataset_path}/data/01_Brandon/00/obj/meshes/00000.obj", "r", buffering=2 ** 10) as fp:
        for line in fp: print(line)
    return 

def main(args):
    ops = (args.operations).split(",")
    if "average_value" in ops:  blend_averge(args)
    if "print_obj" in ops:      print_obj(args)
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="path to the root dir of dataset", default="/home/sage66730/dataset")
    parser.add_argument("--operations", type=str, help="the operations to be performed", default="average_value")

    main(parser.parse_args())