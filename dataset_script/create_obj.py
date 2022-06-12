from asyncio.subprocess import STDOUT
import os
import argparse
import glob
import torch
import shutil
import pickle
import subprocess
from tqdm import tqdm

def clean_subject_obj(subject_path, start):
    sentences = os.listdir(subject_path)
    sentences = sorted(sentences)
    for n, sentence in enumerate(sentences):
        if n < start: continue
        if os.path.isdir(subject_path+"/"+sentence+"/obj"):
            shutil.rmtree(subject_path+"/"+sentence+"/obj")
    print(f"{subject_path} cleaned")

def main(args):
    dataset_path = args.dataset_path
    subjects = os.listdir(dataset_path+"/data")
    if args.target != "*": subjects = (args.target).split(",")
    
    for subject in subjects:
        print(f"processing subject: {subject}")
        subject_path = dataset_path+"/data/"+subject
        if args.ow: clean_subject_obj(subject_path, args.start)
        sentences = os.listdir(subject_path)
        sentences = sorted(sentences)

        tq = tqdm(sentences)
        for n, sentence in enumerate(tq):
            if n < args.start: continue
            #print(f"processing sentence: {sentence}")
            sentence_path = subject_path+"/"+sentence

            if not os.path.isdir(sentence_path+"/auds"):
                print(f"{sentence_path}: audio missing")
                continue
            if os.path.isdir(sentence_path+"/obj"):
                print(f"{sentence_path}: obj exist please --ow")
                continue
            else:
                os.mkdir(sentence_path+"/obj")
                aud_path = glob.glob(f"{sentence_path}/auds/*.wav")[0]

                tq.set_description(f"{sentence_path}")
                subprocess.check_call(["python", f"{args.voca_path}/run_voca.py", 
                                "--tf_model_fname", f"{args.voca_path}/model/gstep_52280.model", 
                                "--ds_fname", f"{args.voca_path}/ds_graph/output_graph.pb", 
                                "--audio_fname", f"{aud_path}", 
                                "--template_fname", f"{args.voca_path}/template/FLAME_sample.ply", 
                                "--condition_idx", "3", 
                                "--out_path", f"{sentence_path}/obj", 
                                "--visualize", "False"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument("--dataset_path", type=str, help="path to the root dir of dataset", default="/home/sage66730/dataset")
    parser.add_argument("--voca_path", type=str, help="path to the root dir of dataset", default="/home/sage66730/voca")
    parser.add_argument("--target", type=str, help="target subject dir to extract", default="*")
    parser.add_argument("--start", type=int, help="number of sentence start from, has to use --target with", default=-1)
    parser.add_argument("--ow", type=bool, help="over write target subject dirs", default=False)

    main(parser.parse_args())
