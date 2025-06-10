import torch
import torch
import numpy as np
import os
#from torch.func import vmap


class DataLoader:
    def __init__(self, block_size, batch_size, vocab_size):
       self.block_size = block_size
       self.batch_size = batch_size
       self.vocab_size = vocab_size
    def get_bubble_batch(self):
      def bubble_pass(idx_raw):
         idx = idx_raw.clone()
         tmp = idx[0].item()
         idx[0] = idx[1]
         idx[1] = tmp
         return idx
         for i in range(self.block_size - 1):
            if idx[i] > idx[i+1] and i % 2 == 0:
               tmp = idx[i].item()
               idx[i] = idx[i + 1]
               idx[i + 1] = tmp
         return idx
      def sort_f(x):
         vals, _ = torch.sort(x)
         return vals

      x = [sort_f(torch.randperm(self.vocab_size)[:self.block_size]) for _ in range(self.batch_size)]
      y = [bubble_pass(idx_raw) for idx_raw in x]
      x = torch.stack(x)
      y = torch.stack(y)
      return x, y
   

