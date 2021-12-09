from monte_carlo_tree_search import MCTS
from game import YukonBoard
from reg_agent import RegAgent
from hur_cy import nikolai
import sys, time, msvcrt
import redis


from collections import deque
import random as rn


import pickle
import torch

import statistics as st
import scipy.stats
import math
import numpy as np
from pathlib import Path
import datetime

from time import sleep


is_CUDA_available = torch.cuda.is_available()
print(f"Checking CUDA avaliability.. {is_CUDA_available}")




piles = 4

print(f"Entered piles {piles}")

if piles == 4:
    dbindex = 2
if piles == 5:
    dbindex = 1
if piles == 6:
    dbindex = 0



logging = True
if logging:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    print("Using logging on tensorboard.")
else:
    print("Warning! Not using logging!")



def new_YukonBoard():
    deck = []
    for i in range(1, 11):
        if i == 10:
            deck.append(16)
        else:
            deck.append(4)

    deck = tuple(deck)
    return YukonBoard(piles=((0,),) * piles, deck=deck, card=None, turn=False, terminal=False)


# load data from previous runs


# Hyperparamaters

use_precompute = True
if use_precompute:
    precompute_cache = dict()
    precompute_cache_uses = dict()
    print("Warning! Using precompute")
else:
    print("Not using precompute!")



external_board = False

if external_board == True:
    from myplayer import Player
    player = Player(piles)
    print("Playing with external board..")
    if player.is_cards_configured == False or player.is_settings_configured == False:
        player.setup()
else:
    print("NOT playing with external board..")

savedir = 123

### Agent
save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
checkpoint = Path('checkpoints/2021-12-05T14-08-33/mario_net_100551.chkpt')
reg_agent = RegAgent(save_dir, checkpoint=checkpoint)

r = redis.Redis(host='127.0.0.1', port=6379, db=0, password='MikkelSterup')




first_board = new_YukonBoard()

e = 1


win_list = []
win_list.append(0)
card_list = []
card_list_moving = deque(maxlen=30)
card_list_moving.append(0)
win_list_moving = deque(maxlen=200)
win_list_moving.append(0)
t = []

cards_trained = 0
prediction_list_moving = deque(maxlen=1000)
prediction_list_moving.append(0)
for _ in range(1):
    prediction_list_moving.append(0)

train = True
local_cache_size = 0
e = 0
last_log_e = 0
last_save_e = 0
loss = 0
reg_agent.nik_rate = 0
r.set('model', pickle.dumps(reg_agent))
boards_to_train = []
while train:
    actual_redis_cache_size = r.llen('train_boards')
    redis_cache_size = min(actual_redis_cache_size, 100)
    e += redis_cache_size
    if redis_cache_size == 0:
        print(f"Total boards: {e}. No boards in redis cache, waiting 10 seconds...")
        sleep(10)
    else:

        print(f"Getting {redis_cache_size} boards.. {actual_redis_cache_size -redis_cache_size} left for next training.")
        for i in range(redis_cache_size):
            boards_to_train.append(pickle.loads(r.lpop('train_boards')))




        if len(boards_to_train) > 100:
            loss += reg_agent.redis_learn(boards_to_train)
            rn.shuffle(boards_to_train)
            loss_sum = loss/(e-last_log_e)
            loss = 0
            last_log_e = e
            writer.add_scalar("Loss sum", torch.FloatTensor([loss_sum]), e)
            r.set('model', pickle.dumps(reg_agent))
            #
            #
            llen_cards = r.llen('cards')
            for i in range(llen_cards):
                card_list_moving.append(int(r.lpop('cards')))
            mean_c = st.mean(card_list_moving)
            writer.add_scalar("Mean/30", torch.FloatTensor([mean_c]), e)

            llen_wr = r.llen('wr')
            for i in range(llen_wr):
                win_list.append(int(r.lpop('wr')))
            mean_w_total = st.mean(win_list)
            mean_w = st.mean(win_list[-200:])
            writer.add_scalar("Win rate, last 200", torch.FloatTensor([mean_w]), e)
            writer.add_scalar("Win rate, total", torch.FloatTensor([mean_w_total]), e)

            writer.add_scalar("Huristics rate", torch.FloatTensor([reg_agent.nik_rate]), e)

            llen_pred = r.llen('pred')
            for i in range(llen_pred):
                prediction_list_moving.append(int(r.lpop('pred')))
            pred_mean = st.mean(prediction_list_moving)
            writer.add_scalar("Prediction mean of last 1000", torch.FloatTensor([pred_mean]), e)

            r.set('model', pickle.dumps(reg_agent))

            if pred_mean > 0.7 and mean_w > 0.2:
                reg_agent.nik_rate = reg_agent.nik_rate - 0.0001

            boards_to_train = []


        if e > last_save_e + 4*30*100:
            last_save_e = e
            reg_agent.save(e)
