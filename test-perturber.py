from ddmodel import PerturberBlock
import torch
weights = torch.load('./saved_ddmodels/itr_1000_checkpoint.pt', map_location=torch.device('cpu'))
weights
#####
model_state_dict = weights['model']
block_size = 8
vocab_size = 128
from ddmodel import GPTConfig
from ddmodel import DDGPT
conf = GPTConfig(block_size=block_size, vocab_size=vocab_size)
####
mymodel = DDGPT(conf)
mymodel.load_state_dict(model_state_dict)
print(mymodel)
print(mymodel.generate(torch.tensor([[4,2,5,10,100,50, 39, 78]])))