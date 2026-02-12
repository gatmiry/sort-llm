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
# Note: checkpoint was trained without positional embeddings (without_pos=True)
config = GPTConfig(block_size=block_size, vocab_size=vocab_size, without_pos=True)
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
    count_perlocation = np.array([0.0 for i in range(block_size)])
    count_perthreshold = []
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
        tmp_count_perlocation = np.array([0.0 for i in range(block_size)])
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
                    tmp_count_perlocation[j-block_size] += 1
                    dist = abs(idx[0,k] - idx[0,j + 1]).item()
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
                count_perlocation[j-block_size] = tmp_count_perlocation[j-block_size]
                average_dist_perlocation[j-block_size] = average_temp_dist
                max_dist_perlocation[j-block_size] = max_dist

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

        count_perthreshold.append(np.mean(tmp_count_perlocation))

        average_max_dist /= block_size
        average_dist /= block_size
        max_dist_perthreshold.append(average_max_dist)
        average_dist_perthreshold.append(average_dist)
        ### added max_dist_perthreshold and average_dist_perthreshold to the list
        clogit_cscore_perthreshold.append(clogit_cscore / block_size)
        clogit_icscore_perthreshold.append(clogit_icscore / block_size)
        iclogit_cscore_perthreshold.append(iclogit_cscore / block_size)
        iclogit_icscore_perthreshold.append(iclogit_icscore / block_size)
    count_perthreshold = np.array(count_perthreshold)
    clogit_cscore_perthreshold = np.array(clogit_cscore_perthreshold)
    clogit_icscore_perthreshold = np.array(clogit_icscore_perthreshold)
    iclogit_cscore_perthreshold = np.array(iclogit_cscore_perthreshold)
    iclogit_icscore_perthreshold = np.array(iclogit_icscore_perthreshold)
    average_dist_perthreshold = np.array(average_dist_perthreshold)
    max_dist_perthreshold = np.array(max_dist_perthreshold)
    return count_perlocation, count_perthreshold, (clogit_cscore_perlocation, clogit_icscore_perlocation, iclogit_cscore_perlocation, iclogit_icscore_perlocation), (clogit_cscore_perthreshold, clogit_icscore_perthreshold, iclogit_cscore_perthreshold, iclogit_icscore_perthreshold), (average_dist_perlocation, max_dist_perlocation), (average_dist_perthreshold, max_dist_perthreshold)


thresholds = [0.01, 0.05, 0.1, 0.15, 0.2]
threshold_index = 0
num_tries = 10
ave_clogit_cscore_perthreshold = np.zeros(len(thresholds))
ave_iclogit_cscore_perthreshold = np.zeros(len(thresholds))
ave_clogit_icscore_perthreshold = np.zeros(len(thresholds))
ave_iclogit_icscore_perthreshold = np.zeros(len(thresholds))
ave_count_perthreshold = np.zeros(len(thresholds))
ave_count_perlocation = np.zeros(block_size)
for counter in range(num_tries):
    count_perlocation, count_perthreshold, lsperlocation, lsperthreshold, am_perlocation, am_perthreshold = get_statistics(thresholds, threshold_index)
    clogit_cscore_perthreshold, clogit_icscore_perthreshold, iclogit_cscore_perthreshold, iclogit_icscore_perthreshold = lsperthreshold
    average_dist_perthreshold, max_dist_perthreshold = am_perthreshold
    clogit_cscore_perlocation, clogit_icscore_perlocation, iclogit_cscore_perlocation, iclogit_icscore_perlocation = lsperlocation
    average_dist_perlocation, max_dist_perlocation = am_perlocation
    
    #print(f'clogit_cscore_perthreshold is {clogit_cscore_perthreshold}')
    ave_clogit_cscore_perthreshold += clogit_cscore_perthreshold
    ave_iclogit_cscore_perthreshold += iclogit_cscore_perthreshold
    ave_clogit_icscore_perthreshold += clogit_icscore_perthreshold
    ave_iclogit_icscore_perthreshold += iclogit_icscore_perthreshold
    ave_count_perthreshold += count_perthreshold
    ave_count_perlocation += count_perlocation

ave_clogit_cscore_perthreshold /= num_tries
ave_iclogit_cscore_perthreshold /= num_tries
ave_clogit_icscore_perthreshold /= num_tries
ave_iclogit_icscore_perthreshold /= num_tries
ave_count_perthreshold /= num_tries
ave_count_perlocation /= num_tries

# Save data to file for later comparison
data_file = 'plots/statistics_data.npz'
np.savez(data_file,
         thresholds=np.array(thresholds),
         ave_clogit_cscore_perthreshold=ave_clogit_cscore_perthreshold,
         ave_iclogit_cscore_perthreshold=ave_iclogit_cscore_perthreshold,
         ave_clogit_icscore_perthreshold=ave_clogit_icscore_perthreshold,
         ave_iclogit_icscore_perthreshold=ave_iclogit_icscore_perthreshold,
         ave_count_perthreshold=ave_count_perthreshold,
         ave_count_perlocation=ave_count_perlocation)
print(f'Data saved to {data_file}')

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
print(f'ave_clogit_cscore_perthreshold is {ave_clogit_cscore_perthreshold}')
print(f'ave_iclogit_cscore_perthreshold is {ave_iclogit_cscore_perthreshold}')
print(f'ave_clogit_icscore_perthreshold is {ave_clogit_icscore_perthreshold}')
print(f'ave_iclogit_icscore_perthreshold is {ave_iclogit_icscore_perthreshold}')
x_pos = np.arange(len(thresholds))
width = 0.6

# First plot: CLogit CScore and ICLogit CScore
plt.figure(figsize=(8, 6))
plt.bar(x_pos, ave_clogit_cscore_perthreshold, width, label='CLogit CScore', color='#1f77b4')
plt.bar(x_pos, ave_iclogit_cscore_perthreshold, width, bottom=ave_clogit_cscore_perthreshold, label='ICLogit CScore', color='#ff7f0e')
plt.xlabel('Threshold')
plt.ylabel('Average Score per Threshold')
plt.title('CScore per Threshold (Stacked)')
plt.xticks(x_pos, thresholds)
plt.legend()
plt.savefig('plots/cscore_perthreshold.png', dpi=150, bbox_inches='tight')
print('Plot saved to plots/cscore_perthreshold.png')
plt.close()

# Second plot: CLogit ICScore and ICLogit ICScore
plt.figure(figsize=(8, 6))
plt.bar(x_pos, ave_clogit_icscore_perthreshold, width, label='CLogit ICScore', color='#2ca02c')
plt.bar(x_pos, ave_iclogit_icscore_perthreshold, width, bottom=ave_clogit_icscore_perthreshold, label='ICLogit ICScore', color='#d62728')
plt.xlabel('Threshold')
plt.ylabel('Average Score per Threshold')
plt.title('ICScore per Threshold (Stacked)')
plt.xticks(x_pos, thresholds)
plt.legend()
plt.savefig('plots/icscore_perthreshold.png', dpi=150, bbox_inches='tight')
print('Plot saved to plots/icscore_perthreshold.png')
plt.close()

# Third plot: Average count per threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds, ave_count_perthreshold, marker='o', linewidth=2, markersize=8, color='#9467bd')
plt.xlabel('Threshold')
plt.ylabel('Average Count per Threshold')
plt.title('Average Count per Threshold')
plt.grid(True, alpha=0.3)
plt.savefig('plots/count_perthreshold.png', dpi=150, bbox_inches='tight')
print('Plot saved to plots/count_perthreshold.png')
plt.close()

# Fourth plot: Average count per location
plt.figure(figsize=(8, 6))
location_indices = np.arange(block_size)
plt.plot(location_indices, ave_count_perlocation, marker='o', linewidth=2, markersize=8, color='#8c564b')
plt.xlabel('Location Index (within block)')
plt.ylabel('Average Count per Location')
plt.title('Average Count per Location')
plt.grid(True, alpha=0.3)
plt.savefig('plots/count_perlocation.png', dpi=150, bbox_inches='tight')
print('Plot saved to plots/count_perlocation.png')
plt.close()