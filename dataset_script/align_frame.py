import os
import csv
import argparse
import glob
import shutil
from tqdm import tqdm

def interpolation(l1, l2):
    
    if l1 == l2:
        return []
    else:
        assert l1 > l2
        m = l1-1
        n = l2-1
        arr = list(range(l2))
        arr = [int( (idx*m)/n ) for idx in arr]
        assert len(arr) == len(set(arr))
        return arr

"""
def align(self, datas, labels):
        assert len(datas) == len(labels)
        data_nums = [len(data) for data in datas]
        label_nums = [len(label) for label in labels]
        selected_idce = interpolation(data_nums, label_nums)

        new_data = []
        new_label = [ frame for sentence in labels for frame in sentence]

        for i, sentence in enumerate(datas):
            selected_idx = selected_idce[i]
            new_sentence = [ frame for j, frame in enumerate(sentence) if j in selected_idx]
            new_data += new_sentence

        assert len(new_data) == len(new_label)
        return new_data, new_label
"""

def main(args):
    dataset_path = args.dataset_path
    subjects = os.listdir(dataset_path+"/data")
    if args.target != "*": subjects = (args.target).split(",")
    
    for subject in subjects:
        subject_path = dataset_path+"/data/"+subject
        sentences = os.listdir(subject_path)
        sentences = sorted(sentences)

        tq = tqdm(sentences)
        tq.set_description(f"{subject}")
        for sentence in tq:
            objs = glob.glob(f"{dataset_path}/data/{subject}/{sentence}/obj/meshes/*.obj")
            data_num = len(objs)

            csv_path = glob.glob(f"{dataset_path}/label/{subject}/{sentence}/M*.csv")
            fp = open(csv_path[0], newline='')
            cdata = list(csv.reader(fp))
            label_num = len(cdata)-1

            new_label_idx = interpolation(label_num, data_num)
            with open(f"{dataset_path}/label/{subject}/{sentence}/aligned.csv", "w", newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(cdata[0])
                for idx in new_label_idx:
                    csvwriter.writerow(cdata[idx+1])
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="path to the root dir of dataset", default="/home/sage66730/dataset")
    parser.add_argument("--target", type=str, help="target subject dir to extract", default="*")

    main(parser.parse_args())
