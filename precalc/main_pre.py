import scipy.stats
import math
import numpy as np
from pathlib import Path
import datetime
from collections import deque
import statistics as st
import pickle
import torch
import redis
import sys
import os
import io
from multiprocessing import Process, cpu_count
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)
from monte_carlo_tree_search import MCTS
from game import YukonBoard
from reg_agent import RegAgent


def new_YukonBoard():
    piles = 4
    deck = []
    for i in range(1, 11):
        if i == 10:
            deck.append(16)
        else:
            deck.append(4)

    deck = tuple(deck)
    return YukonBoard(piles=((0,),) * piles, deck=deck, card=None, turn=False, terminal=False)



def simulate(sim_board):
    while True:
        sim_board = (sim_board)

        if sim_board.terminal or sum(sim_board.deck) == 0:
            if sum(sim_board.deck) == 0:
                return True
            else:
                return False
        sim_board = sim_board.make_move(0)  # draw card


def get_move(board, future_board):
    piles = 4
    winner = None
    for i in range(piles):
        if board.make_move(i) == future_board:
            winner = i
            break

    assert winner is not None, "Winner is none - Future board not a child of current board?"
    return winner


def gameloop():
    piles = 4
    r = redis.Redis(host='82.211.216.32', port=6379, db=0, password='MikkelSterup')

    is_CUDA_available = torch.cuda.is_available()
    print(f"Checking CUDA avaliability.. {is_CUDA_available}")

    piles = 4

    use_precompute = False

    first_board = new_YukonBoard()

    searches_def = 3000

    now = datetime.datetime.now()

    do_100_of_each = True

    print(f"Do 100 of each option before starting proper MCTS? {do_100_of_each}")

    print("")

    print("Please make sure all settings are correct")
    e = 0
    external_board = False
    while True:
        data = []
        e += 1
        reg_agent = pickle.loads(r.get('cpu_agent'))
        reg_agent.use_cuda = False
        print(f"Hur rate is: {reg_agent.nik_rate}")

        time_list = 53 * [0]
        its_list = 53 * [0]
        board = first_board

        board = board.make_move(0)  # draw a a card
        temp_won_game_list = []

        hur_solved = False

        huristic_cards = 0
        precomputed_cards = 0
        one_option_cards = 0
        prediction_list_moving = []

        c = 0

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
                        for _ in range(int(r.get('pregame'))):
                            tree.do_rollout(board)
                        for _ in range(searches_def):
                            tree.do_rollout(board)

                            if _ % int(r.get('rosetti')) == 0:
                                score, winner = tree.choose(board)
                                test_list.append(max(score))
                                n0 = len(test_list)
                                if len(test_list) > 4:
                                    S = st.stdev(test_list)
                                    T = scipy.stats.t.interval(alpha, len(test_list) - 1, loc=0, scale=1)[-1]
                                    h0 = T * (S / math.sqrt(n0 - 1))

                                    sort_score = score[-4:]
                                    sort_score = set(sort_score)

                                    last_item = 0
                                    this_item = 0
                                    for each in sort_score:
                                        last_item = this_item
                                        this_item = each

                                    h = (this_item - last_item) / (2)  # More MCTS
                                    if h == 0:
                                        # Avoid div by zero
                                        continue

                                    if all([x == 0 for x in score]) and _ > 750:
                                        winner = get_move(board, tree.huristic(board))
                                        print(f"All scores zero - using huristics and braking training.")

                                    N = n0 * (h0 / h) ** 2

                                    if N < _:
                                        print(
                                            f"Rosetti says: Number of samples is appropriate after {N}. Actual samples run is {_}")
                                        break

                            if _ == searches_def - 1:
                                print(f"Max searches reached at N = {searches_def}. Rosetti suggests {N}.")

                        score, winner = tree.choose(board)
                        if all([x == 0 for x in score]):
                            winner = get_move(board, tree.huristic(board))
                            print(f"All scores zero - using huristics")

                        predictions = []
                        for each in range(1, piles + 1):
                            future_board = board.make_move(each - 1)
                            future_score = score[each]

                            prediction = reg_agent.act(future_board)
                            data.append((future_board, future_score))
                            predictions.append(prediction)

                        pred_card = np.argmax(predictions)

                        prediction = reg_agent.act(future_board)

                        print(" ")
                        print(f"Scores vs predictions:         Score of current state:  {score[0]}")
                        print(f"                               Score of prediction was: {reg_agent.act(board)}*")
                        print(" ")
                        score = score[-piles:]
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
                    # mm_hits , mm_miss , _, _ = board.make_move.cache_info()

                    print(f"Memorization information: ")
                    print(f"Items in tree: {len(tree.children)}")
                    print(f"    Huristics: Hits: {hur_hits}, Miss: {hur_miss}, Rate: {hur_hits / (hur_hits + hur_miss)}")
                    print(
                        f"    reg_agent.act: Hits: {act_hits}, Miss: {act_miss}, Rate: {act_hits / (act_hits + act_miss)}")
                    # print(f"    reg_agent.act: Hits: {mm_hits}, Miss: {mm_miss}, Rate: {mm_hits / (mm_hits + mm_miss)}")

                    pred_mean = st.mean(prediction_list_moving)

                    pred_loss = 0
                    if not hur_solved:
                        for i in range(4):
                            pred_loss += abs(score[i] - predictions[i])



            board = board.make_move(winner)

            if board.terminal or sum(board.deck) == 0:
                past = now
                now = datetime.datetime.now()
                dt = now - past

                print(" ")
                print("----------------------")
                print(f"Current time {now} Delta_t {dt}")
                board.show(c, e)

                learn = True
                if learn:
                    to_redis = pickle.dumps([data, c, huristic_cards, one_option_cards, precomputed_cards, prediction_list_moving])
                    r.rpush('datalist', to_redis)
                else:
                    loss_sum = 0




                # log results

                if sum(board.deck) == 0:
                    win = 1
                else:
                    win = 0


                break

            else:

                board = board.make_move(0)  # draw a a card
if __name__ == '__main__':
    #gameloop()
    num_workers=cpu_count()
    print(f"Starting worker count: {num_workers}")
    processes = []
    for _ in range(num_workers):
        p = Process(target=gameloop)
        p.start()
        processes.append(p)

