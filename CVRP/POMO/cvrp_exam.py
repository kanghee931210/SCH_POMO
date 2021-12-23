##########################################################################################
# Machine Environment Config
import matplotlib.pyplot as plt

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys
from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
from utils.utils import *
import torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from CVRPTester import CVRPTester as Tester


##########################################################################################
# parameters
aug_factor = 1
# if not tester_params['augmentation_enable']:
#     aug_factor = tester_params['aug_factor']
# else:
#     aug_factor = 1

env_params = {
    'problem_size': 50,
    'pomo_size': 50,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/saved_CVRP20_model',  # directory path of pre-trained model and log files saved.
        'epoch': 2000,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 100*1000,
    'test_batch_size': 10000,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 5000,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test_cvrp_n20',
        'filename': 'run_log'
    }
}

batch_size = 1
##########################################################################################
# main


env = Env(**env_params)
model = Model(**model_params)
model_load = tester_params['model_load']
# checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
# checkpoint_fullname = "./result/saved_CVRP50_model/checkpoint-2000.pt"
checkpoint_fullname = "./result/saved_CVRP50_model/checkpoint-2000.pt"
checkpoint = torch.load(checkpoint_fullname, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
with torch.no_grad():
    env.load_problems(1, aug_factor)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

# plot
#
plot_param = env_params['problem_size']

time_h = 4*10*24
time_m = 60

depot = np.array(reset_state.depot_xy).reshape(2)
node = np.array(reset_state.node_xy).reshape(plot_param,2)

node_x_ori = node[:,0]
node_y_ori = node[:,1]

node_x = node[:,0]#*time_h
node_y = node[:,1]#*time_m
plt.scatter(depot[0],depot[1], color = 'r' )
plt.scatter(node_x_ori,node_y_ori)
plt.annotate(f"start",xy = (depot[0],depot[1]))
for i in range(len(node)):
    # plt.annotate(f"{i}th point",xy = (node_x[i],node_y[i]))
    plt.annotate(f"{node_x[i]}h{node_y[i]}m", xy=(node_x[i], node_y[i]))
    # plt.annotate(f"{i}th point {lenth[0][i]:.2f}",xy = (node_x[i],node_y[i]))
# plt.title(f'{total_len}')
plt.show()
# POMO Rollout
###############################################
state, reward, done = env.pre_step()
while not done:
    selected, _ = model(state)
    # shape: (batch, pomo)

    state, reward, done = env.step(selected)

# for i in range(len(reset_state_x)):
#     plt.annotate(f"{i}th point {lenth[0][i]:.2f}",xy = (reset_state_x[i],reset_state_y[i]))
# plt.title(f'{total_len}')
# plt.show()
#
# Return
###############################################
aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
# shape: (augmentation, batch, pomo)

max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
index_r = np.argmax(reward)
# shape: (augmentation, batch)
no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
# shape: (batch,)
aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value
resize = env.selected_node_list.shape[2]
ress = np.array(env.selected_node_list).reshape(plot_param,resize)
ind = ress[index_r]
all_spot = np.array(env.depot_node_xy).reshape(plot_param+1,2)

test_list = list(ind)
result_list = list(filter(lambda x: test_list[x] == 0, range(len(test_list))))

plot_list = []
for i in range(len(result_list)):
    if i ==0:
        pass
    else:
        a = test_list[result_list[i-1]+1:result_list[i]]
        if len(a) > 0:
            a.append(a[0])
            plot_list.append(a)

# cluster code
total_x = []
total_y = []
for k in range(len(plot_list)):
    cord_x = []
    cord_y = []
    for i in plot_list[k]:
        cord_x.append(all_spot[i][0])
        cord_y.append(all_spot[i][1])
    total_x.append(np.array(cord_x))#*time_h)
    total_y.append(np.array(cord_y))#*time_m)

ax = plt.subplot(1,1,1)
for j in range(len(total_x)):
    plt.plot(total_x[j],total_y[j], 'o-')
plt.annotate(f"start",xy = (depot[0],depot[1]))

for i in range(len(node)):
    plt.annotate(f"{node_x[i]}h{node_y[i]}m", xy=(node_x[i], node_y[i]))
    # plt.annotate(f"{i}th point",xy = (node_x[i],node_y[i]))
# plt.title(f'{max_pomo_reward:.2f}')
plt.title(f'{aug_score:.2f}')
plt.show()
# print(env.selected_node_list)

# all code
cord_x_ = []
cord_y_ = []

for i in ind:
    cord_x_.append(all_spot[i][0])
    cord_y_.append(all_spot[i][1])
plt.plot(np.array(cord_x_)*time_h,np.array(cord_y_)*time_m)
plt.annotate(f"start",xy = (depot[0],depot[1]))
for i_ in range(len(node)):
    plt.annotate(f"{node_x[i_]}h{node_y[i_]}m", xy=(node_x[i_], node_y[i_]))
    # plt.annotate(f"{i}th point",xy = (node_x[i],node_y[i]))
# plt.title(f'{max_pomo_reward:.2f}')
plt.title(f'{aug_score:.2f}')
plt.show()
