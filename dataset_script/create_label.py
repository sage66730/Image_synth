import os
import glob
import shutil
import argparse
import subprocess

def main(args):
    subjects = os.listdir("./raw_data")
    if args.target != "*": subjects = (args.target).split(",")
    if args.ow: 
        for subject in subjects: 
            if os.path.isdir("./label/"+subject): shutil.rmtree("./label/"+subject)

    for subject in subjects:
        if os.path.isdir("./label/"+subject):
            print(f"{subject}: subject exist please --ow")
            continue
        else:
            print(f"processing subject: {subject}")
            os.mkdir("./label/"+subject)
            sentences = glob.glob("./raw_data/"+subject+"/*/*.csv")
            sentences = sorted(sentences)
            for sentence in sentences:
                print(f"processing sentence: {sentence}")
                sentence_name = sentence.split("/")[-2]
                os.mkdir("./label/"+subject+"/"+sentence_name)
                
                shutil.copy2(sentence, f"./label/{subject}/{sentence_name}")

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--target", help="target subject dir to extract", type=str, default="*")
    a.add_argument("--ow", help="over write target subject dirs", type=bool, default=False)
    args = a.parse_args()
    
    main(args)