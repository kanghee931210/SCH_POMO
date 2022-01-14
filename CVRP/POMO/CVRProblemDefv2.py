import pandas as pd
import torch
import numpy as np
import os
import random

def get_random_problems(batch_size, problem_size):

    depot_xy = torch.zeros(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_x = torch.zeros(size=(batch_size, problem_size,1))
    node_y = torch.randint(1, 1440, size=(batch_size, problem_size,1))

    node_xy = torch.cat((node_x,node_y),dim = 2)
    # shape: (batch, problem, 2)

    demand_scaler = 480
    node_demand = torch.randint(40, 60, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand

def SCH_problems(day):
    csv = pd.read_csv("./SCHdataset.csv")
    day = int(day)

    depot_xy = torch.zeros(size=(1, 1, 2))
    dataset = csv[(csv['DEPARRDT'] == day)]# & (csv['hour2min'] > 540)]
    node_x = torch.Tensor(list(dataset['hour']))[None, :, None]
    node_y = torch.Tensor(list(dataset['min']))[None, :,None]
    demand = torch.Tensor(list(dataset['working_time']))[None,:]


    exam_y = 60 * node_x + node_y
    exam_x = torch.zeros(size=node_x.shape)

    demand_scaler = 480
    node_demand = demand / float(demand_scaler)

    node_xy = torch.cat((node_x, node_y), dim=2)
    node_exam = torch.cat((exam_x, exam_y), dim=2)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand , demand, node_exam



def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data
