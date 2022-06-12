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
            if os.path.isdir("./data/"+subject): shutil.rmtree("./data/"+subject)

    for subject in subjects:
        if os.path.isdir("./data/"+subject):
            print(f"{subject}: subject exist please --ow")
            continue
        else:
            print(f"processing subject: {subject}")
            os.mkdir("./data/"+subject)
            sentences = glob.glob("./raw_data/"+subject+"/*/*.mov")
            sentences = sorted(sentences)
            for sentence in sentences:
                print(f"processing sentence: {sentence}")
                sentence_name = sentence.split("/")[-2]
                file_name = sentence.split("/")[-1].split(".")[0]
                os.mkdir("./data/"+subject+"/"+sentence_name)
                os.mkdir("./data/"+subject+"/"+sentence_name+"/auds")
            
                subprocess.run(["ffmpeg", "-y", "-i", sentence, f"./data/{subject}/{sentence_name}/auds/{file_name}.wav"], 
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--target", help="target subject dir to extract", type=str, default="*")
    a.add_argument("--ow", help="over write target subject dirs", type=bool, default=False)
    args = a.parse_args()
    
    main(args)