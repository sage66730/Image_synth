import os
import glob
import shutil
import argparse
import subprocess

def main(args):
    subjects = os.listdir(f"{args.dataset_path}/raw_data")
    if args.target != "*": subjects = (args.target).split(",")
    if args.ow: 
        for subject in subjects: 
            if os.path.isdir(f"{args.dataset_path}/label/"+subject): shutil.rmtree(f"{args.dataset_path}/label/"+subject)

    for subject in subjects:
        if os.path.isdir(f"{args.dataset_path}/label/"+subject):
            print(f"{subject}: subject exist please --ow")
            continue
        else:
            print(f"processing subject: {subject}")
            os.mkdir(f"{args.dataset_path}/label/"+subject)
            sentences = glob.glob(f"{args.dataset_path}/raw_data/"+subject+"/*/*.csv")
            sentences = sorted(sentences)
            for sentence in sentences:
                print(f"processing sentence: {sentence}")
                sentence_name = sentence.split("/")[-2]
                os.mkdir(f"{args.dataset_path}/label/"+subject+"/"+sentence_name)
                
                shutil.copy2(sentence, f"{args.dataset_path}/label/{subject}/{sentence_name}")

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--dataset_path", type=str, help="path to the root dir of dataset", default="/home/sage66730/dataset")
    a.add_argument("--target", help="target subject dir to extract", type=str, default="*")
    a.add_argument("--ow", help="over write target subject dirs", type=bool, default=False)
    args = a.parse_args()
    
    main(args)