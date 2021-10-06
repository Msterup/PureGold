import torch
import torch.optim as optim

from neural import Net
from game import YukonBoard

from collections import deque
import random as rn
import itertools
from functools import lru_cache




class Agent:
    def __init__(self, save_dir, checkpoint=None):
        self.save_dir = save_dir
        self.net = Net()
        self.loss_fn = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=0.00001, momentum=0.9)
        self.memory = deque(maxlen=100000)

        self.use_cuda = False
        self.save_every = 10
        self.prediction_rate = 0

        if checkpoint:
            self.load(checkpoint)

    @lru_cache(maxsize=1000)
    def act(self, board):
        """Given a state, choose an epsilon-greedy action"""
        perm_act = True

        if perm_act:
            # find all permutations of the state
            permutations = itertools.permutations(board.piles)
            unique_permutations = []
            for each in permutations:
                unique_permutations.append(each)
            unique_permutations = set(unique_permutations)

            act_list = []
            for each in unique_permutations:
                act_list.append(YukonBoard(each, deck=board.deck, card=board.card, turn=True, terminal=False))

            act_total = 0
            if len(act_list) > 2:
                act_list = rn.sample(act_list, 2)
            for each in act_list:
                output = self.net.forward(each.tensorize())
                act_total += output.item()

            act_mean = act_total/len(act_list)

            return act_mean

        else:
            return self.net.forward(board.tensorize()).item()


    def cache(self, board, score):
        """Add the experience to memory"""
        """Experience: Board, score"""
        permutations = itertools.permutations(board.piles)
        unique_permutations = []
        for each in permutations:
            unique_permutations.append(each)
        unique_permutations = set(unique_permutations)

        for each in unique_permutations:
            self.memory.append((YukonBoard(each, deck=board.deck, card=board.card, turn=True, terminal=False), score))


        return

    def recall(self):
        """Sample experiences from memory"""

        return rn.sample(self.memory, 3000)

    def learn(self):
        sum_loss = 0
        experiences = self.recall()
        for input, label in experiences:
            self.optimizer.zero_grad()
            output = self.net.forward(input.tensorize())
            label = torch.tensor([label])
            loss = self.loss_fn(output, label)
            loss.backward()
            self.optimizer.step()
            sum_loss += loss.item()

        return sum_loss



    def save(self, e):
        """Saves the nural net in a  """
        save_path = self.save_dir / f"mario_net_{int(e // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
            ),
            save_path
        )
        print(f"NeuralNet saved to {save_path} at step {e}")

        pass

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path}")
        self.net.load_state_dict(state_dict)



    def tensorize_label(self, label):
        if type(label) == int:
            #tensor = [[0, 0, 0, 0]]
            #tensor[0][label] = 1
            tensor = torch.LongTensor([label])

            return tensor
