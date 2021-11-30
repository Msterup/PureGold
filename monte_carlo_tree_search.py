"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import numpy as np

from hur_cy import nikolai
import random as rn
from functools import lru_cache


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, agent, exploration_weight=500):
        self.agent = agent
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.F = defaultdict(float)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.nik_rate = 1

    @lru_cache(1000*3)
    def huristic(self, board):
        fq = board.find_quick()
        if fq is None:
            return nikolai(board)
        else:
            return board.make_move(fq)


    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")


        MCTS_result = []
        winner = None

        def score(n):
            if self.N[n] == 0:
                return 0
            return self.Q[n] / self.N[n]

        MCTS_result.append(score(node))
        for i in range(len(node.piles)):
            curr = node.make_move(i)
            MCTS_result.append(score(curr))

        winner = np.nanargmax(MCTS_result[-len(node.piles):])

        return MCTS_result, winner



    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)


    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                if node.turn:
                    fq = node.find_quick()
                    if fq is not None:
                        n = node.make_move(fq)
                    else:
                        if rn.random() > self.nik_rate:
                            n = self.net_select(unexplored, node)
                        else:
                            nik_play = node.make_move(self.huristic(node))

                            if nik_play in unexplored:
                                n = nik_play
                            else:
                                n = self.net_select(unexplored, node)


                else:
                    n = node.find_random_child()

                path.append(n)
                return path

            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"

        while True:
            if node.is_terminal():
                return node.reward()
            if node.turn:
                fq = node.find_quick()
                if fq is not None:
                    node = node.make_move(fq)
                else:
                    net_total_prediction = False
                    if net_total_prediction:
                        return self.agent.act(node)
                    if rn.random() > self.agent.nik_rate:
                        node = self.net_select(node.find_children(), node)
                    else:
                        nik_play = self.huristic(node)
                        node = node.make_move(nik_play)
            else:
                node = node.find_random_child()


    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward


    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        #assert all(n in self.children for n in self.children[node])
        if node.turn:

            log_N_vertex = math.log(self.N[node])


            def uct(n):
                "Upper confidence bound for trees"

                wr = self.Q[n] / self.N[n]
                ln = self.exploration_weight * math.sqrt(log_N_vertex / self.N[n])

                if n.is_terminal() == True and self.N[n] > 10:
                    terminal = 0
                else:
                    terminal = 1

                return (self.F[n]+wr+ln)*terminal


            return max(node.find_children(), key=uct)
        else:
            return node.find_random_child()

    def net_select(self, nodes, original):
        scores = []
        for each in nodes:

            if not each.is_terminal:
                score = 0
            else:
                score = self.agent.act(each)
            scores.append((score, each))

        def getmax(x):
            return x[0]

        winner = max(scores, key=getmax)
        if winner[0] == 0:
            winner = original.make_move(self.huristic(original))
        else:
            winner = winner[1]

        return winner



class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """


    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self, current_goal):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True