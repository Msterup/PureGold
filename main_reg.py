from monte_carlo_tree_search import MCTS
from game import YukonBoard
from reg_agent import RegAgent
from hur_cy import nikolai
import sys, time, msvcrt


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
import redis


r = redis.Redis(host='10.250.13.234', port=6379, db=0, password='MikkelSterup')

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

use_precompute = False
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
checkpoint = Path('checkpoints/2022-03-10T20-18-12/mario_net_10.chkpt')
reg_agent = RegAgent(save_dir, checkpoint=checkpoint)

r = redis.Redis(host='127.0.0.1', port=6379, db=0, password='MikkelSterup')



first_board = new_YukonBoard()

e = 1


win_list = []
card_list = []
card_list_moving = deque(maxlen=30)
win_list_moving = deque(maxlen=200)
t = []

cards_trained = 0
prediction_list_moving = deque(maxlen=1000)
for _ in range(1):
    prediction_list_moving.append(0)

current_goal = 20
searches_def = 3000

save_data = False
if save_data:
    sample_list = [1, 2, 3]
    file_name = "won_games_1105.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()
    master_won_game_list = []

def simulate(sim_board):
    while True:
        sim_board = (sim_board)

        if sim_board.terminal or sum(sim_board.deck) == 0:
            if sum(sim_board.deck) == 0:
                return True
            else:
                return False
        sim_board = sim_board.make_move(0) #draw card

def get_move(board, future_board):
    winner = None
    for i in range(piles):
        if board.make_move(i) == future_board:
            winner = i
            break

    assert winner is not None, "Winner is none - Future board not a child of current board?"
    return winner


now = datetime.datetime.now()

do_100_of_each = False

print(f"Do 100 of each option before starting proper MCTS? {do_100_of_each}")


print("")
print("Please make sure all settings are correct")
input("Press Enter to continue...")


for e in range(5000):


    time_list = 53 * [0]
    its_list = 53 * [0]

    if external_board == True:
        player.reset_real_game()





    board = first_board

    if external_board:
        sleep(1)
        drawn_card = player.draw_card()
        board = board.make_move(drawn_card)
    else:
        board = board.make_move(0) # draw a a card
    temp_won_game_list = []

    hur_solved = False

    huristic_cards = 0
    precomputed_cards = 0
    one_option_cards = 0



    c = 0

    reg_agent.num_chached = 0
    tree = None
    while True:
        if tree == None:
            tree = MCTS(reg_agent)

        c += 1
        past = now
        now = datetime.datetime.now()
        dt = now - past

        print(" ")
        print("----------------------")
        print(f"Current time {now} Delta_t {dt}")
        board.show(c, e)
        winner = None

        if use_precompute:
            board_flat = board.flatten()
            if r.exists(str(board_flat)):
                winner = int(r.get(board_flat))
                precomputed_cards += 1

        if winner is not None:
            print(f"This board was already computed - winner is {winner}")

        else:
            s = 0
            sim_wins = []
            fq = board.find_quick()
            if fq is not None:
                winner = fq
                print("Find quick option selected")
                one_option_cards += 1
                tree = None
                score = [1001]

            else:
                # Enable or disable hur_solve
                while False:
                        s += 1
                        sim_wins.append(simulate(board))
                        if all(sim_wins) == True and s == 99:
                                hur_solved = True
    
    
                if hur_solved == True:
                    print("Solved by 100 huristic simulations")
                    winner = get_move(board, tree.huristic(board))
                    score = [1001]
                    huristic_cards += 1

                if not hur_solved:
                    if do_100_of_each:
                        for each in board.find_children():
                            if each.is_terminal:
                                for _ in range(100):
                                    tree.do_rollout(each)

                    # prob good: alpha = 0.9999995
                    alpha = 0.999999995
                    test_list = []
                    for _ in range(searches_def):
                        tree.do_rollout(board)

                        if _ % 15 == 0:
                            score, winner = tree.choose(board)
                            test_list.append(max(score))
                            n0 = len(test_list)
                            if len(test_list) > 4:
                                S = st.stdev(test_list)
                                T = scipy.stats.t.interval(alpha, len(test_list)-1, loc=0, scale=1)[-1]
                                h0 = T*(S/math.sqrt(n0-1))

                                sort_score = score[-4:]
                                sort_score = set(sort_score)

                                last_item = 0
                                this_item = 0
                                for each in sort_score:
                                    last_item = this_item
                                    this_item = each

                                h = (this_item-last_item)/(2) #More MCTS
                                if h == 0:
                                    # Avoid div by zero
                                    continue

                                if all([x == 0 for x in score]) and _ > 750:
                                    winner = get_move(board, tree.huristic(board))
                                    print(f"All scores zero - using huristics and braking training.")

                                N = n0*(h0/h)**2

                                if N < _:
                                    print(f"Rosetti says: Number of samples is appropriate after {N}. Actual samples run is {_}")
                                    break

                        if _ == searches_def-1:
                            print(f"Max searches reached at N = {searches_def}. Rosetti suggests {N}.")

                    score, winner = tree.choose(board)
                    if all([x == 0 for x in score]):
                        winner = get_move(board, tree.huristic(board))
                        print(f"All scores zero - using huristics")

                    predictions = []
                    for each in range(1,piles+1):
                        future_board = board.make_move(each-1)
                        future_score = score[each]
                        reg_agent.cache(future_board, future_score)
                        prediction = reg_agent.act(future_board)
                        predictions.append(prediction)
                        reg_agent.num_cached += 1

                    pred_card = np.argmax(predictions)




                    prediction = reg_agent.act(future_board)




                    print(" ")
                    print(f"Scores vs predictions:         Score of current state:  {score[0]}")
                    print(f"                               Score of prediction was: {reg_agent.act(board)}*")
                    print(" ")
                    score = score[-4:]
                    print(score)
                    print(predictions)
                    print(" ")
                    print(f"Huristic option was   {get_move(board, tree.huristic(board))}")
                    print(f"Neural net option was {pred_card}")
                    print(f"Winning option was    {winner}")
                    print("")






                    if pred_card == winner:
                        prediction_list_moving.append(True)
                        print("Neural net was right! (Right pile.)")
                    elif score[pred_card] == score[winner]:
                        prediction_list_moving.append(True)
                        print("Neural net was right! (Wrong pile, but equal.)")
                    else:
                        prediction_list_moving.append(False)
                        print("Neural net was wrong!")


                hur_hits, hur_miss, _, _ = tree.huristic.cache_info()
                act_hits, act_miss, _, _ = reg_agent.act.cache_info()
                #mm_hits , mm_miss , _, _ = board.make_move.cache_info()

                print(f"Memorization information: ")
                print(f"Items in tree: {len(tree.children)}")
                print(f"    Huristics: Hits: {hur_hits}, Miss: {hur_miss}, Rate: {hur_hits/(hur_hits+hur_miss)}")
                print(f"    reg_agent.act: Hits: {act_hits}, Miss: {act_miss}, Rate: {act_hits/(act_hits+act_miss)}")
                #print(f"    reg_agent.act: Hits: {mm_hits}, Miss: {mm_miss}, Rate: {mm_hits / (mm_hits + mm_miss)}")

                pred_mean = st.mean(prediction_list_moving)
                
                if logging:
                    writer.add_scalar("Prediction mean of last 1000", torch.FloatTensor([pred_mean]),
                                  cards_trained)
                pred_loss = 0
                if not hur_solved:
                    for i in range(4):
                        pred_loss += abs(score[i]-predictions[i])

                if logging:
                    writer.add_scalar("Prediction loss", torch.FloatTensor([pred_loss]),
                                  cards_trained)

                cards_trained += 1




        if external_board == True:
            player.get_real_action(winner, board, drawn_card)

        if use_precompute and c < 15:
            precompute_cache[board] = winner
            if board not in precompute_cache_uses:
                precompute_cache_uses[board] = 0


        board = board.make_move(winner)




        if board.terminal or sum(board.deck) == 0:
            past = now
            now = datetime.datetime.now()
            dt = now-past

            print(" ")
            print("----------------------")
            print(f"Current time {now} Delta_t {dt}")
            board.show(c, e)

            learn = True
            if learn:
                loss_sum = reg_agent.learn()
            else:
                loss_sum = 0
            print(f"Game ended at card {c}. Current goal was {current_goal}. Loss was {loss_sum}")

            card_list_moving.append(c)
            mean_c = st.mean(card_list_moving)

            if not current_goal == 52:
                if current_goal < mean_c+2:
                    current_goal += 1

            #log results

            if sum(board.deck) == 0:
                print("win")
                win_list_moving.append(1)
                win_list.append(1)
            else:
                win_list_moving.append(0)
                win_list.append(0)
            mean_w = st.mean(win_list_moving)
            mean_w_total = st.mean(win_list)

            print(f"Current mean is {mean_c}")
            print(f"Current winr is {mean_w}")

            if logging:
                writer.add_scalar("Card", torch.FloatTensor([c]), e)
                writer.add_scalar("Mean/30", torch.FloatTensor([mean_c]), e)
                writer.add_scalar("Win rate, last 200", torch.FloatTensor([mean_w]), e)
                writer.add_scalar("Win rate, total", torch.FloatTensor([mean_w_total]), e)

                writer.add_scalar("Huristic Cards", torch.FloatTensor([huristic_cards]), e)
                writer.add_scalar("Precomputed cards", torch.FloatTensor([precomputed_cards]), e)
                writer.add_scalar("One option cards", torch.FloatTensor([one_option_cards]), e)
                writer.add_scalar("Loss sum", torch.FloatTensor([loss_sum]), e)
                writer.add_scalar("Num experiences", torch.FloatTensor([len(reg_agent.memory)]), e)

                writer.add_scalar("Huristics rate", torch.FloatTensor([reg_agent.nik_rate]), e)
            if pred_mean > 0.50:
                reg_agent.nik_rate = reg_agent.nik_rate -0.01  # Hyper parameter

            if e % reg_agent.save_every == 0:
                reg_agent.save(e)

            if external_board:
                sleep(5)

            break

        else:
            if external_board:
                sleep(1)
                drawn_card = player.draw_card()
                board = board.make_move(drawn_card)
            else:
                board = board.make_move(0)  # draw a a card
# All games now played

open_file = open(file_name, "wb")
pickle.dump(master_won_game_list, open_file)
open_file.close()
print(f"Number of samples in won games list: {len(master_won_game_list)}")

