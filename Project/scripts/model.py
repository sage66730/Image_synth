import torch
from torch import nn

class ObjModel1(torch.nn.Module):
    # Naive model that maps everything straight to blendshape

    def __init__(self, dim_in, dim_out, dim_h1 = 100):
        super(ObjModel1, self).__init__()
        self.dim_in = dim_in

        self.linear1 = nn.Linear(dim_in, dim_h1)
        self.bn = nn.BatchNorm1d(dim_h1)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(dim_h1, dim_out)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        x = self.linear1(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

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