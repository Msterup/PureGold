import random as rn
import game_globals as globals
from functools import lru_cache

import torch




import datetime

from collections import namedtuple
from monte_carlo_tree_search import Node


_YB = namedtuple("YukonBoard", "piles deck card turn terminal")

class YukonBoard(_YB, Node):
    @lru_cache(maxsize=10000)
    def find_children(board, iso=False):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        if board.turn:
            return {
                board.make_move(i, iso) for i in range(0, len(board.piles))
            }
        else:
            return {
                board.make_move(i+1, iso) for i in range(0, 10) if board.deck[i] > 0
            }

    def find_quick(board):

        if len(board.find_children(iso=True)) == 1:
            all_same = True
        else:
            all_same = False

        winners = []
        for i in range(len(board.piles)):
            test_board = board.make_move(i)
            winners.append(test_board.is_terminal())

        num_terminal = winners.count(False)
        return all_same, num_terminal

    def find_random_child(board):
        if board.turn:
            return board.make_move(rn.randint(0, (len(board.piles)-1)))
        else:
            return board.make_move(0)

    @lru_cache(maxsize=10000)
    def find_numbered_child(board, action):
        #if board.turn:
            #return board.make_move(action)
        #else:
        return board.make_move(action)

    @lru_cache(maxsize=10000)
    def reward(board):
        if not board.terminal:
            raise RuntimeError(f"reward called on nonterminal board {board}")

        if board.terminal:
            if sum(x for x in board.deck) == 0:
                max = 0
                for each in board.piles:
                    if len(each) > 1:
                        check = each[1]
                    else:
                        check = each[0]
                    if check > max:
                        max = check
                if max > 21:
                    return (52 -sum(x for x in board.deck))/10
                else:
                    return 1000
            else:
                return (52 -sum(x for x in board.deck))/10


    def is_terminal(board):
        return board.terminal

    def is_exhausted(board):
        return board.exhausted


    def make_move(board, index, iso=False):



        if board.turn:
            piles, is_terminal = board.evaluate(index, iso)
            card = None
            deck = board.deck
            if sum(x for x in board.deck) == 0:
                is_terminal = True



        else:
            is_terminal = board.terminal
            deck, card = board.draw(index)
            piles = board.piles



        prob = 1





        turn = not board.turn
        is_exhausted = is_terminal

        return YukonBoard(piles, deck, card, turn, is_terminal)

    @lru_cache(maxsize=10000)
    def evaluate(board, index, iso):
        is_terminal = False
        pile = list(board.piles[index])
        n_pile = []
        for elem in pile:
            n_pile.append(elem+board.card)
            if board.card == 1:
                n_pile.append(elem + board.card+10)
                break
        if any(x == 21 for x in n_pile):
            n_pile = [0]
        if n_pile[0] > 21:
            is_terminal = True
        else:
            n_pile = [x for x in n_pile if x <= 21]
        n_piles = list(board.piles)
        n_piles[index] = tuple(n_pile)

        #if iso:
        n_piles.sort()
        piles = tuple(n_piles)


        return piles, is_terminal

    def draw(board, index):
        if index == 0:
            card = rn.choices((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), weights=board.deck, k=1)[0]
        else:
            card = index

        deck = list(board.deck)
        deck[card-1] = deck[card-1] - 1

        deck = tuple(deck)

        return deck, card

    def show(board, c, e):
        now = datetime.datetime.now()

        print(" ")
        print("----------------------")
        print("Current time " + now.strftime("%Y-%m-%d %H:%M:%S"))
        print(" ")
        print(board.piles)
        print(" ")
        print(f"Current card: |{board.card}| Card no: {c} Episode no: {e}")
        print(" ")
        print(" 1  2  3  4  5  6  7  8  9  10")
        print(board.deck)


    def tensorize(board):
        """
        to_tensor = []
        for each in board.piles:
            if len(each) > 1:
                to_tensor.append(1)
            else:
                to_tensor.append(0)

        for each in board.piles:
            to_tensor.append(max(each))

        to_tensor.append(board.card)
        state = to_tensor + list(board.deck)

        return torch.FloatTensor(state)
        """
        tensor = [0] * 52
        tensor[sum(board.deck)] = 1

        for each in board.piles:
            if len(each) == 1:
                # only 1 pile:
                item = [0] * 33
                item[0] = 1
                tensor += item
                item = [0] * 33
                item[each[0]] = 1
                tensor += item
            else:
                item = [0] * 33
                item[each[0]] = 1
                tensor += item
                item = [0] * 33
                item[each[1]] = 1
                tensor += item

        for each in range(0,9):
            item = [0] * 5
            item[board.deck[each]] = 1
            tensor += item

        item = [0] * 17
        item[board.deck[9]] = 1
        tensor += item

        return torch.Tensor(tensor)





    @lru_cache(maxsize=10000)
    def give_result(self, pile, card, result):
        if card == 1: # check 10
            if pile + card + 10 == result or pile + card == result:
                return 1
        else:
            if pile + card == result:
                return 1

    @lru_cache(maxsize=10000)
    def expert(self):
        pileList = list(self.piles)
        beforeProbs21 = [0] * len(pileList)

        for i in range(len(pileList)):
            for val in pileList[i]:
                for each in range(len(self.deck)):
                    if self.give_result(val, self.card, 21) == 1:
                        beforeProbs21[i] += self.deck[each]

        afterProbs21 = [0] * len(pileList)

        for i in range(len(pileList)):
            for val in pileList[i]:
                for each in range(len(self.deck)):
                    if self.give_result(val, self.card, 21) == 1:
                        afterProbs21[i] += self.deck[each]

        diffProbs21 = [0] * len(pileList)

        for i in range(len(beforeProbs21)):
            diffProbs21[0] = afterProbs21[0] - beforeProbs21[0]

        if max(diffProbs21) > 0:
            return diffProbs21.index(max(diffProbs21))

        beforeProbs11 = [0] * len(pileList)

        for i in range(len(pileList)):
            for val in pileList[i]:
                for each in range(len(self.deck)):
                    if val + each + 1 == 11:
                        beforeProbs11[i] += self.deck[each]

        afterProbs11 = [0] * len(pileList)

        for i in range(len(pileList)):
            for val in pileList[i]:
                for each in range(len(self.deck)):
                    if val + self.card + each + 1 == 11:
                        afterProbs11[i] += self.deck[each]

        diffProbs11 = [0] * len(pileList)

        for i in range(len(beforeProbs21)):
            diffProbs21[0] = afterProbs21[0] - beforeProbs21[0]

        diffdiff = [0] * len(pileList)

        for i in range(len(beforeProbs21)):
            diffdiff[0] = (diffProbs11[0] * diffProbs21[0]) + diffProbs11[0] * diffProbs21[0]

        # start choice
        # make 21

        for i in range(len(pileList)):
            for val in pileList[i]:
                if self.give_result(val, self.card, 21) == 1:
                    return i
        # make 11
        for i in range(len(pileList)):
            for val in pileList[i]:
                if self.give_result(val, self.card, 11) == 1:
                    return i

        if max(diffProbs21) > 0:
            return diffProbs21.index(max(diffProbs21))

        if max(diffProbs21) > 0 and self.deck[0] >= 1:
            return diffProbs11.index(max(diffProbs11))

        for i in range(len(pileList)):
            for val in pileList[i]:
                if self.give_result(val, self.card, self.card) == 1:
                    return i

        return diffdiff.index(max(diffdiff))

        # return 0

    @lru_cache(maxsize=10000)
    def flatten(self):
        bits = []
        # board_states:
        piles = []
        for each in self.piles:
            if each[-1] > 22:
                piles.append(22)
            else:
                piles.append(each[-1])

        board_state = globals.possible_board_states[piles[0]][piles[1]][piles[2]][piles[3]]
        flat = format(int(board_state), '028b')
        for pile in self.piles:
            if len(pile) == 1:
                flat = flat + (format(0, '01b'))
            else:
                flat = flat + (format(1, '01b'))
        if self.card is None:
            flat = flat + (format(0, '04b'))

        else:
            flat = flat + (format(self.card, '04b'))
        # deck states for deck 1-9
        deck_state = globals.possible_deck_states[self.deck[0]][self.deck[1]][self.deck[2]][self.deck[3]][self.deck[4]][
            self.deck[5]][self.deck[6]][self.deck[7]][self.deck[8]]
        flat = flat + (format(deck_state, '021b'))
        # remember 10'ers
        flat = flat + (format(self.deck[9], '05b'))
        return int(flat, 2)