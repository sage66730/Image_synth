from re import A
import torch
from torch import nn

class ObjModel1(torch.nn.Module):
    # Naive model that maps everything straight to blendshape

    def __init__(self, dim_in, dim_out, dim_h1 = 100):
        super(ObjModel1, self).__init__()
        self.dim_in = dim_in

        self.linear1 = nn.Linear(dim_in, dim_h1*2)
        self.bn1 = nn.BatchNorm1d(dim_h1*2)
        self.activation1 = nn.Sigmoid()

        self.linear3 = nn.Linear(dim_h1*2, dim_h1)
        self.bn3 = nn.BatchNorm1d(dim_h1)
        self.activation3 = nn.ReLU()

        self.linear2 = nn.Linear(dim_h1, dim_out)
        self.activation2 = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.activation3(x)
        x = self.linear2(x)
        x = self.activation2(x)
        return x

class ObjMJModel1(torch.nn.Module):
    # 00~13: EYE
    # 14~17: JAW
    # 18~40: MOTHE
    # 41~45: BROW
    # 46~48: CHEEK
    # 49~50: NOSE
    # 51   : TONGUE
    # 52~60: MOTION
    
    def __init__(self, dim_in, dim_out, dim_h1 = 100):
        super(ObjMJModel1, self).__init__()
        self.dim_in = dim_in

        # for mouth
        self.m_linear1 = nn.Linear(415*3, dim_h1)
        self.m_bn1 = nn.BatchNorm1d(dim_h1)
        self.m_activation1 = nn.Sigmoid()

        self.m_linear2 = nn.Linear(dim_h1, 23)
        self.m_activation2 = nn.ReLU()

        # for jaw
        self.j_linear1 = nn.Linear(108*3, dim_h1)
        self.j_bn1 = nn.BatchNorm1d(dim_h1)
        self.j_activation1 = nn.Sigmoid()

        self.j_linear2 = nn.Linear(dim_h1, 4)
        self.j_activation2 = nn.ReLU()

        # for all the blendshape
        self.a_linear1 = nn.Linear(dim_in, dim_h1*2)
        self.a_bn1 = nn.BatchNorm1d(dim_h1*2)
        self.a_activation1 = nn.Sigmoid()

        self.a_linear3 = nn.Linear(dim_h1*2, dim_h1)
        self.a_bn3 = nn.BatchNorm1d(dim_h1)
        self.a_activation3 = nn.ReLU()

        self.a_linear2 = nn.Linear(dim_h1, 34)
        self.a_activation2 = nn.ReLU()

    def forward(self, x):
        m, j, a = x[0], x[1], x[2]
        #print(m.shape, j.shape, a.shape)

        m = m.view(-1, 415*3)
        m = self.m_linear1(m)
        m = self.m_bn1(m)
        m = self.m_activation1(m)
        m = self.m_linear2(m)
        m = self.m_activation2(m)

        j = j.view(-1, 108*3)
        j = self.j_linear1(j)
        j = self.j_bn1(j)
        j = self.j_activation1(j)
        j = self.j_linear2(j)
        j = self.j_activation2(j)

        a = a.view(-1, self.dim_in)
        a = self.a_linear1(a)
        a = self.a_bn1(a)
        a = self.a_activation1(a)
        a = self.a_linear3(a)
        a = self.a_bn3(a)
        a = self.a_activation3(a)
        a = self.a_linear2(a)
        a = self.a_activation2(a)

        result = torch.cat((a[:,:14], m, j, a[:,14:34]), 1)
        return result

class ExpModel2(torch.nn.Module):
    # Model that process exp and jaw seperately
    # Consider jaw pose affect every key

    def __init__(self, dim_exp, dim_jaw):
        super(ExpModel2, self).__init__()

        self.linear_exp = nn.Sequential(nn.Linear(dim_exp, 100,),
                                        nn.BatchNorm1d(100),
                                        nn.ReLU())
        
        self.linear_jaw = nn.Sequential(nn.Linear(dim_jaw, 100,),
                                        nn.BatchNorm1d(100),
                                        nn.ReLU())
        
        self.linear = nn.Linear(100, 52)

    def forward(self, exp, jaw):
        exp = self.linear_exp(exp)
        jaw = self.linear_jaw(jaw)
        result = exp + jaw
        result = self.linear(result)
        return result