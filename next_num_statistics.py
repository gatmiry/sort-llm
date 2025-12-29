from model_tbyt_3 import GPT, GPTConfig
import torch
import os
import numpy as np
#block_size = 8
#vocab_size = 128
block_size = 32
vocab_size = 128
instance_num = 1
device = 'cuda'
config = GPTConfig(block_size=block_size, vocab_size=vocab_size)
model = GPT(config)
#model_state_dict = torch.load(os.path.join(os.getcwd(), f'saved_models/tbyt_1head_2_itr:{itr_num}_checkpoint_old.pt'), map_location=device)['model']
model_state_dict = torch.load(os.path.join(os.getcwd(), f'saved_models/dec28_tbyt_without-pos-embedding_n_embd:64_1head_layers:2_vocab_size:128_itr:60000_checkpoint.pt'), map_location=device)['model']
#model_state_dict = torch.load('./saved_models/tbyt_b64_v2048_embd16_1head_2_itr:20000_checkpoint.pt', map_location=device)['model']
model.load_state_dict(model_state_dict)
model.to(device=device)

def get_batch(batch_size=1):
   def cat_sorted_tensor(x):
      vals, _ = torch.sort(x)
      return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0)
   #x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_size)[:block_size]) for _ in range(batch_size)])
   x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_size)[:32]) for _ in range(batch_size)])
   return x

threshold = 0.05

def get_statistics(thresholds, threshold_index):
    clogit_cscore_perlocation = np.array([0.0 for i in range(block_size)])
    clogit_icscore_perlocation = np.array([0.0 for i in range(block_size)])
    iclogit_cscore_perlocation = np.array([0.0 for i in range(block_size)])
    iclogit_icscore_perlocation = np.array([0.0 for i in range(block_size)])
    clogit_cscore_perthreshold = []
    clogit_icscore_perthreshold = []
    iclogit_cscore_perthreshold = []
    iclogit_icscore_perthreshold = []
    
    average_dist_perlocation = np.array([0.0 for i in range(block_size)])
    max_dist_perlocation = np.array([0.0 for i in range(block_size)])
    max_dist_perthreshold = []
    average_dist_perthreshold = []
    for t, threshold in enumerate(thresholds):
        clogit_cscore = 0
        clogit_icscore = 0
        iclogit_cscore = 0
        iclogit_icscore = 0
        average_max_dist = 0.0
        average_dist = 0.0
        idx = get_batch(1)
        print(f'idx shape is {idx.shape}')
        idx = idx.to(device=device)
        logits, loss = model(idx)
        is_correct = torch.where(torch.argmax(logits[0 , block_size: 2 * block_size, :], dim=1) == idx[0, block_size + 1:], torch.ones(block_size).to(device), torch.zeros(block_size).to(device))
        print(f'is_correct is {is_correct}')
        all_candidates = []
        
        candidates = [[] for j in range(block_size, 2*block_size + 1)]
        is_largest_score_correct = [None for j in range(block_size, 2*block_size + 1)]
        for j in range(block_size, 2*block_size):
            max_score = float('-inf')
            max_score_num = -1
            max_dist = 0.0
            average_temp_dist = 0.0
            dist = 0.0
            num_dist = 0
            for k in range(0, 2*block_size + 1):
                score = model.transformer.h[0].c_attn.attn[j,k].item()
                if score >= threshold:
                    dist = abs(idx[0,k] - idx[0,j + 1])
                    if dist > max_dist:
                        max_dist = dist
                    num_dist += 1
                    average_temp_dist += dist
                    if score > max_score:
                        #print('im here')
                        max_score = score
                        max_score_num = idx[0,k].item()
                        #print('max_score_num is ', max_score_num)
                    candidates[j-block_size].append((k,idx[0,k].item()))
                    all_candidates.append((k,idx[0,k].item()))
            average_max_dist += max_dist
            average_temp_dist /= num_dist
            average_dist += average_temp_dist
            #print('max_score_num is ', max_score_num, ' idx[0,j + 1].item() is ', idx[0,j + 1].item())
            is_largest_score_correct[j-block_size] = (max_score_num == idx[0,j + 1].item())    

            if t == threshold_index:
                average_dist_perlocation[j-block_size] = average_temp_dist
                max_dist_perlocation[j-block_size] = average_max_dist

            if is_correct[j-block_size] == 1.0 and is_largest_score_correct[j-block_size] == True:
                if t == threshold_index:
                    clogit_cscore_perlocation[j-block_size] = 1.0
                clogit_cscore += 1
            elif is_correct[j-block_size] == 1.0 and is_largest_score_correct[j-block_size] == False:
                if t == threshold_index:
                    clogit_icscore_perlocation[j-block_size] = 1.0
                clogit_icscore += 1
            elif is_correct[j-block_size] == 0.0 and is_largest_score_correct[j-block_size] == True:
                if t == threshold_index:
                    iclogit_cscore_perlocation[j-block_size] = 1.0
                iclogit_cscore += 1
            elif is_correct[j-block_size] == 0.0 and is_largest_score_correct[j-block_size] == False:
                if t == threshold_index:
                    iclogit_icscore_perlocation[j-block_size] = 1.0
                iclogit_icscore += 1

            print(f'candidates for position {j} with number {idx[0,j].item()} and is_correct {is_correct[j-block_size] == 1.0} are {candidates[j-block_size]} \n \
                    is_largest_score_correct is {is_largest_score_correct[j-block_size]}\n')    
            if is_largest_score_correct[j-block_size] == False:
                print(f'scores for position {j} are {[(num, model.transformer.h[0].c_attn.attn[j,k].item()) for k, num in candidates[j-block_size]]}\n\n\n')

        average_max_dist /= block_size
        average_dist /= block_size
        max_dist_perthreshold.append(average_max_dist)
        average_dist_perthreshold.append(average_dist)
        ### added max_dist_perthreshold and average_dist_perthreshold to the list
        clogit_cscore_perthreshold.append(clogit_cscore / block_size)
        clogit_icscore_perthreshold.append(clogit_icscore / block_size)
        clogit_cscore_perthreshold.append(iclogit_cscore / block_size)
        iclogit_icscore_perthreshold.append(iclogit_icscore / block_size)
    clogit_cscore_perlocation = [clogit_cscore_perlocation[i] / len(thresholds) for i in range(block_size)]
    clogit_icscore_perlocation = [clogit_icscore_perlocation[i] / len(thresholds) for i in range(block_size)]
    iclogit_cscore_perlocation = [iclogit_cscore_perlocation[i] / len(thresholds) for i in range(block_size)]
    iclogit_icscore_perlocation = [iclogit_icscore_perlocation[i] / len(thresholds) for i in range(block_size)]
    return (clogit_cscore_perlocation, clogit_icscore_perlocation, iclogit_cscore_perlocation, iclogit_icscore_perlocation), (clogit_cscore_perthreshold, clogit_icscore_perthreshold, iclogit_cscore_perthreshold, iclogit_icscore_perthreshold), (average_dist_perlocation, max_dist_perlocation), (average_dist_perthreshold, max_dist_perthreshold)


