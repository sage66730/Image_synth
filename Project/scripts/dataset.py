import csv
import glob
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

def interpolation(l1, l2):
    assert len(l1) == len(l2)

    selected_idce = []
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            selected_idce.append(list(range(l2[i])))
        else:
            assert l1[i] > l2[i]
            m = l1[i]-1
            n = l2[i]-1
            arr = list(range(l2[i]))
            arr = [int( (idx*m)/n ) for idx in arr]
            assert len(arr) == len(set(arr))
            selected_idce.append(arr)
    return selected_idce

def load_obj(fp):
    temp = []
    for line in fp:
        line= line.split()
        if line[0] == "v":
            odata = [float(line[1]), float(line[2]), float(line[3])]
            temp.append(odata)
        else: break
    return temp

def load_csv(fp):
    cdata = csv.reader(fp)
    temp = []
    for idx, row in enumerate(cdata):
        if idx != 0:
            tensor = [float(num) for num in row[2:]]
            temp.append(torch.Tensor(tensor))
    return temp  

class ObjDataset(Dataset):
    def __init__(self, dataset_path, subjects):
        self.dataset_path = dataset_path
        self.datas, self.labels = self.load_data_label(dataset_path, subjects)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):

        with open(self.datas[idx], "r", buffering=2 ** 10) as fp:
            odata = load_obj(fp)
            odata = torch.Tensor(odata)
    
        subject  = self.datas[idx].split("/")[-5]
        sentence = self.datas[idx].split("/")[-4]
        number   = int(self.datas[idx].split("/")[-1].split(".")[0])

        with open(f"{self.dataset_path}/label/{subject}/{sentence}/aligned.csv", newline='') as fp:
            cdata = load_csv(fp)
        
        if len(cdata[number]) == 0:
            fp = open("error.txt", "a") 
            fp.write(f"{self.dataset_path}/label/{subject}/{sentence}/aligned.csv   {number}\n")
            return odata, torch.Tensor([0]*61)
        
        return [subject, sentence, number], odata, cdata[number]

    def load_data_label(self, dataset_path, subjects):
        if subjects == ["*"]:
            obj_paths = glob.glob(dataset_path+"/data/*/*/obj/meshes/*.obj")
            csv_paths = glob.glob(dataset_path+"/label/*/*/aligned.csv")
        else:
            subjects = subjects.split(",")
            obj_paths = []
            for subject in subjects:
                obj_paths += glob.glob(dataset_path+"/data/"+subject+"/*/obj/meshes/*.obj")
            csv_paths = []
            for subject in subjects:
                csv_paths += glob.glob(dataset_path+"/label/"+subject+"/*/aligned.csv")
        obj_paths = sorted(obj_paths)
        csv_paths = sorted(csv_paths)
        
        return obj_paths, csv_paths

class ExpDataset(Dataset):
    def __init__(self, dataset_path, subjects):
        self.datas, self.labels = self.load_data_label(dataset_path, subjects)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.datas[idx], self.lables[idx]

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

    def load_data_label(self, dataset_path, subjects):
        if subjects == ["*"]:
            exps_paths = glob.glob(dataset_path+"/data/*/*/exps/*.npy")
            csv_paths = glob.glob(dataset_path+"/label/*/*/*.csv")
        else:
            exps_paths = []
            for subject in subjects:
                exps_paths += glob.glob(dataset_path+"/data/"+subject+"/*/exps/exps.pkl")
            csv_paths = []
            for subject in subjects:
                csv_paths += glob.glob(dataset_path+"/label/"+subject+"/*.csv")
        exps_paths = sorted(exps_paths)
        csv_paths = sorted(csv_paths)
        
        datas = []
        for exps_path in exps_paths:
            #TBD
            with open(exps_path,"rb") as fp:
                ndata = np.load(fp)
                datas.append(ndata)
        
        labels = []
        for csv_path in csv_paths:
            with open(csv_path, newline='') as fp:
                cdata = csv.reader(fp)
                temp = []
                for idx, row in enumerate(cdata):
                    if idx != 0:
                        tensor = [float(num) for num in row[2:]]
                        temp.append(torch.Tensor(tensor))
                labels.append(temp)  

        datas, labels = self.align(datas, labels)
        return datas, labels

if __name__ == "__main__":
    #test = ExpDataset(dataset_path="/home/sage66730/dataset", subjects=["*"])
    test = ObjDataset(dataset_path="/home/sage66730/dataset", subjects=["*"])
    d, l = test.__getitem__(0)
    print(len(d),len(l))
    print(type(d),type(l))