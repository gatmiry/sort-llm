import torch
import torch.nn as nn

mymodel = nn.Linear(3,4)
ipt = torch.ones(2,3)
out = mymodel(ipt)
print(f'out: {out}')
print(f'cuda: {torch.cuda.is_available()}')