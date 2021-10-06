from monte_carlo_tree_search import MCTS
from game import YukonBoard

from collections import deque
import statistics as st

import pickle
import torch
from hur_cy import nikolai
from myplayer import Player
import statistics as st
import scipy.stats
import math

from time import sleep



piles = 4

print(f"Entered piles {piles}")

if piles == 4:
    dbindex = 2
if piles == 5:
    dbindex = 1
if piles == 6:
    dbindex = 0

player = Player(piles)


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()



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

external_board = True

if external_board == True:
    if player.is_cards_configured == False or player.is_settings_configured == False:
        player.setup()

#logger = MetricLogger

savedir = 123

#piles = int(input("Enter piles: "))


first_board = new_YukonBoard()

e = 1

win_list = []
win_list_moving = []
card_list = []
card_list_moving = deque(maxlen=30)
win_list_moving = []
t = []

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
        sim_board = sim_board.make_move(nikolai(sim_board))

        if sim_board.terminal or sum(sim_board.deck) == 0:
            if sum(sim_board.deck) == 0:
                return True
            else:
                return False
        sim_board = sim_board.make_move(0)

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
    while True:
        tree = MCTS()
        c += 1
        board.show(c, e)
        s = 0
        sim_wins = []
        all_same, non_terminal_move, num_non_terminal = board.find_quick()
        if all_same == True:
            board.make_move(0)
            score = 1
            winner = 0
            print("Only one option avaliable")
            one_option_cards += 1

        elif num_non_terminal == 1:

            winner = non_terminal_move
            board.make_move(winner)
            score = 1
            print("Only one option avaliable")
            one_option_cards += 1

        else:

            while s < 100 and hur_solved == False:
                    s += 1
                    sim_wins.append(simulate(board))
                    if all(sim_wins) == True and s == 99:
                            hur_solved = True


            if hur_solved == True:
                print("Solved by 100 huristic simulations")
                #winner = board.expert()
                winner = nikolai(board)
                score = 1
                huristic_cards += 1

            else:
                # prob good: alpha = 0.9999995
                alpha = 0.999999995
                test_list = []
                for _ in range(searches_def):
                    tree.do_rollout(board)

                    if _ % 50 == 0:
                        score, winner = tree.choose(board)
                        test_list.append(max(score))
                        n0 = len(test_list)
                        if len(test_list) > 10:
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

                            h = (this_item-last_item)/2

                            N = n0*(h0/h)**2

                            if N < _:
                                print(f"Rosetti says: Number of samples is appropriate after {N}. Actual samples run is {_}")
                                break

                            if _ == searches_def-1:
                                print(f"Max searches reached at N = {searches_def}. Rosetti suggests {N}.")



                score, winner = tree.choose(board)



        #net_pred = net.predict_show(board.tensorize())
        #print(f"Net predictions: {net_pred}")
        print(f"Winning option is {winner}")

        #state = board.tensorize()
        #agent.cache(state, torch.FloatTensor(score))
        #temp_won_game_list.append((board, winner))

        if external_board == True:
            player.get_real_action(winner, board, drawn_card)
        board = board.make_move(winner)

        print(score)


        if board.terminal or sum(board.deck) == 0:
            board.show(c, e)
            print(f"Game ended at card {c}. Current goal was {current_goal}")

            card_list_moving.append(c)
            mean_c = st.mean(card_list_moving)

            if not current_goal == 52:
                if current_goal < mean_c+2:
                    current_goal += 1

            #log results

            if sum(board.deck) == 0:
                print("win")
                win_list_moving.append(1)
                for each in temp_won_game_list:
                    master_won_game_list.append(each)
            else:
                win_list_moving.append(0)
            mean_w = st.mean(win_list_moving)

            print(f"Current mean is {mean_c}")
            print(f"Current winr is {mean_w}")

            writer.add_scalar("Card", torch.FloatTensor([c]), e)
            writer.add_scalar("Mean/30", torch.FloatTensor([mean_c]), e)
            writer.add_scalar("Win rate", torch.FloatTensor([mean_w]), e)

            writer.add_scalar("Huristic Cards", torch.FloatTensor([huristic_cards]), e)
            writer.add_scalar("Precomputed cards", torch.FloatTensor([precomputed_cards]), e)
            writer.add_scalar("One option cards", torch.FloatTensor([one_option_cards]), e)

            #if e % 10 == 0:
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

