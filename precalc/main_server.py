from collections import deque
import statistics as st

import time
import pickle
import torch
import redis
from copy import deepcopy

import sys
import os
import io
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)
from reg_agent import RegAgent

r = redis.Redis(host='82.211.216.32', port=6379, db=0, password='MikkelSterup')

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

r = redis.Redis(host='82.211.216.32', port=6379, db=0, password='MikkelSterup')
r = redis.Redis(host='127.0.0.1', port=6379, db=0, password='MikkelSterup')

### Agent
load_from = "File"
if load_from == "Redis":
    reg_agent = pickle.loads(r.get('cpu_agent'))
elif load_from == "File":
    reg_agent = RegAgent()
    reg_agent.net.load_state_dict(torch.load("precalc/4_files_final.pt"))
else:
    reg_agent = RegAgent()
    r.set('cpu_agent', pickle.dumps(reg_agent))

reg_agent.to_cuda()
r.set('pregame', 1000)
r.set('rosetti', 30)
print("")
print("Please make sure all settings are correct")



win_list = []
card_list = []
card_list_moving = deque(maxlen=30)
win_list_moving = deque(maxlen=200)

hur_list = []
one_option_list = []
precomputed_list = []
prediction_list = [] = deque(maxlen=1000)
trained_its = 0

e = 0
card_sum = 0
while True:
    num_items_in_list = r.llen('datalist')
    print(f"Num items in list: {num_items_in_list}")
    if num_items_in_list == 0:
        time.sleep(10)
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        e += 1
        data, card, huristic_cards, one_option_cards, precomputed_cards, prediction = pickle.loads(r.rpop('datalist'))
        reg_agent.cache(data)
        if len(reg_agent.memory) < 32:
            continue
        loss_sum, its = reg_agent.learn()
        print(f"Got {its}")
        trained_its += its

        for each in prediction:
            prediction_list.append(each)
        pred_mean = st.mean(prediction_list)

        if card == 52:
            win_list_moving.append(1)
            win_list.append(1)
        else:
            win_list_moving.append(0)
            win_list.append(0)

        card_list.append(card)
        mean_c = st.mean(card_list)
        mean_w = st.mean(win_list_moving)
        mean_w_total = st.mean(win_list)

        card_sum += card

        if logging:
            writer.add_scalar("Card", torch.FloatTensor([card]), e)
            writer.add_scalar("Mean/30", torch.FloatTensor([mean_c]), e)
            writer.add_scalar("Win rate, last 200", torch.FloatTensor([mean_w]), e)
            writer.add_scalar("Win rate, total", torch.FloatTensor([mean_w_total]), e)

            writer.add_scalar("Huristic Cards", torch.FloatTensor([huristic_cards]), e)
            writer.add_scalar("Precomputed cards", torch.FloatTensor([precomputed_cards]), e)
            writer.add_scalar("One option cards", torch.FloatTensor([one_option_cards]), e)
            writer.add_scalar("Loss sum", torch.FloatTensor([loss_sum]), e)
            writer.add_scalar("Num experiences", torch.FloatTensor([trained_its]), e)
            writer.add_scalar("Prediction, last 1000", torch.FloatTensor([pred_mean]), card_sum)

            if pred_mean > 0.75:
                reg_agent.nik_rate = reg_agent.nik_rate - 0.001  # Hyper parameter
            writer.add_scalar("Huristics rate", torch.FloatTensor([reg_agent.nik_rate]), e)

        if e % 5 == 0:
            #torch.save(reg_agent.net.state_dict(), "temp_model")
            #cpu_agent = RegAgent()
            #cpu_agent.nik_rate = reg_agent.nik_rate
            #cpu_agent.net.load_state_dict(torch.load("temp_model", map_location=torch.device('cpu')))
            #r.set('cpu_agent', pickle.dumps(cpu_agent))
            r.set('cpu_agent', pickle.dumps(reg_agent))
            print("Net has been sent to redis!")




