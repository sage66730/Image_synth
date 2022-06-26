import os
import argparse

def main(args):
    subjects = os.listdir(f"{args.dataset_path}/raw_data")

    for subject in subjects:
        sentences = os.listdir(f"{args.dataset_path}/raw_data/{subject}")
        if sorted(sentences)[0] == "00": continue

        sentences = sorted(sentences, key= lambda x: int(x.split("_")[-1]))
        for n, sentence in enumerate(sentences):
            os.rename(f"{args.dataset_path}/raw_data/{subject}/{sentence}",f"{args.dataset_path}/raw_data/{subject}/{n:02d}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="path to the root dir of dataset", default="/home/sage66730/dataset")

    main(parser.parse_args())
