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
            return [subject, sentence, number], odata, torch.Tensor([0]*61)
        
        return [subject, sentence, number], odata, cdata[number]

    def load_data_label(self, dataset_path, subjects):
        obj_paths = []
        for subject in subjects:
            obj_paths += glob.glob(dataset_path+"/data/"+subject+"/*/obj/meshes/*.obj")
        csv_paths = []
        for subject in subjects:
            csv_paths += glob.glob(dataset_path+"/label/"+subject+"/*/aligned.csv")
        obj_paths = sorted(obj_paths)
        csv_paths = sorted(csv_paths)
        
        return obj_paths, csv_paths

class ObjMJDataset(Dataset):
    def __init__(self, dataset_path, subjects):
        self.dataset_path = dataset_path
        self.datas, self.labels = self.load_data_label(dataset_path, subjects)
        self.mouth = [805, 806, 807, 825, 826, 827, 861, 862, 948, 949, 950, 957, 958, 959, 986, 987, 1000, 1001, 1002, 1047, 1048, 1049, 1172, 1173, 1174, 1380, 1381, 1382, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1395, 1396, 1397, 1398, 1402, 1403, 1446, 1447, 1448, 1468, 1469, 
        1473, 1474, 1475, 1477, 1478, 1483, 1503, 1504, 1505, 1506, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1545, 1546, 1547, 1548, 1549, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1581, 1582, 1583, 1591, 1592, 1593, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 
        1609, 1612, 1614, 1615, 1616, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1646, 1647, 1648, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1686, 1687, 1691, 1692, 1693, 1694, 1695, 1696, 1706, 1707, 1708, 1709, 1710, 1711, 1744, 1747, 1788, 1789, 1841, 1842, 2051, 2099, 2103, 2119, 2120, 2193, 2256, 2780, 2881, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 
        2908, 2909, 2911, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2971, 2982, 2983, 2984, 2985, 2986, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049, 
        3050, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3153, 3154, 3155, 3182, 3362, 3503, 3604, 3796, 3797, 3798, 3799, 3805, 3829, 3830, 3859, 3862, 3863, 
        3866, 3868, 3870, 3873, 3874, 3883, 3886, 3889, 3890, 3900, 3901, 3902, 3908, 3910, 3919, 3920, 3921, 3922, 3923, 3924, 5016]
    
        self.jaw = [807, 948, 949, 950, 1596, 1623, 1633, 1634, 1635, 3063, 3091, 3093, 3094, 3151, 3152, 3507, 3508, 3509, 3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3600, 3602, 3603, 3604, 3609, 3610, 3611, 3616, 3620, 3621, 3622, 3623, 3624, 
        3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3650, 3653, 3654, 3655, 3659, 3660, 3662, 3764, 3765, 3766, 3767, 3768, 3769, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3778, 3779, 3780, 3781, 3782, 3783, 3784, 3785, 3786, 3794, 3795, 3796, 3797, 3806, 3807, 3808, 3809, 3810, 3816, 3817, 3818, 3819, 3820, 3821, 3822, 3825, 3829, 3844, 3852, 3853]


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):

        with open(self.datas[idx], "r", buffering=2 ** 10) as fp:
            odata = load_obj(fp)
            odata = torch.Tensor(odata)
        
        m = odata[self.mouth]
        j = odata[self.jaw]
    
        subject  = self.datas[idx].split("/")[-5]
        sentence = self.datas[idx].split("/")[-4]
        number   = int(self.datas[idx].split("/")[-1].split(".")[0])

        with open(f"{self.dataset_path}/label/{subject}/{sentence}/aligned.csv", newline='') as fp:
            cdata = load_csv(fp)
        
        if len(cdata[number]) == 0:
            fp = open("error.txt", "a") 
            fp.write(f"{self.dataset_path}/label/{subject}/{sentence}/aligned.csv   {number}\n")
            return [subject, sentence, number], [m, j, odata], torch.Tensor([0]*61)
        
        return [subject, sentence, number], [m, j, odata], cdata[number]

    def load_data_label(self, dataset_path, subjects):
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
    test = ObjMJDataset(dataset_path="/home/sage66730/dataset", subjects=["*"])
    _, d, l = test.__getitem__(0)
    print(len(d[0]),len(d[1]),len(d[2]))
    print(type(d[0]),type(d[1]),type(d[2]))