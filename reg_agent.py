import torch
import torch.optim as optim

from neural import ResNet, ResidualBlock
from game import YukonBoard

from collections import deque
import random as rn
import itertools
from functools import lru_cache
import math
from tqdm import tqdm

class RegAgent:
    def __init__(self):
        self.net = ResNet(ResidualBlock, 5)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.0002, betas=(0.09, 0.0999), eps=1e-08, weight_decay=0.0005, amsgrad=False)
        self.memory = []
        self.save_every = 10
        self.nik_rate = 1.15

    def to_cuda(self):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda = torch.device('cuda')
            self.net.to(self.cuda)
        else:
            print("Cant use cuda on this one buddy")

    @lru_cache(maxsize=1000*3)
    def act(self, board, grad=True):
        """Given a state, choose an epsilon-greedy action"""
        self.net.eval()
        perm_act = False

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
            if self.use_cuda:
                return self.net.forward(board.tensorize().cuda()).item()
            else:
                return self.net.forward(board.tensorize()).item()


    def cache(self, tree):
        if self.use_cuda:
            for N, key in tqdm(enumerate(tree.N)):
                if N >= 100 and (not key.turn):
                    self.memory.append((key.tensorize().cuda(), torch.tensor([tree.Q[key] / tree.N[key]]).cuda()))
        else:
            for N, key in tqdm(enumerate(tree.N)):
                if N >= 100 and (not key.turn):
                    self.memory.append((key.tensorize(), torch.tensor([tree.Q[key] / tree.N[key]])))

        return

    def recall(self):
        """Sample experiences from memory"""
        if len(self.memory) < 32:
            return None
        rn.shuffle(self.memory)
        minibatch = []
        for _ in range(32):
            minibatch.append(self.memory.pop())
        return minibatch


    def learn(self):
        its = math.floor(len(self.memory)/32)
        print(f"Learning... ")
        self.net.train(True)
        minibatch = self.recall()
        num_trained = 0
        sum_loss = 0
        with tqdm(total=its) as pbar:
            while minibatch is not None:
                input, label = map(torch.stack, zip(*minibatch))
                self.optimizer.zero_grad()
                output = self.net.forward(input)
                loss = self.loss_fn(output, label)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                num_trained += 32
                minibatch = self.recall()
                pbar.update(1)

        print(f"Number of trained experiences: {num_trained}")

        return sum_loss/num_trained, its*32

    def redis_learn(self, data):
        sum_loss = 0
        for input, label in data:
            self.optimizer.zero_grad()
            output = self.net.forward(input.tensorize())
            label = torch.tensor([label])
            loss = self.loss_fn(output, label)
            loss.backward()
            self.optimizer.step()
            sum_loss += loss.item()


        return sum_loss



    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path}")
        self.net.load_state_dict(state_dict)
        self.nik_rate = ckp.get('nik_rate')



    def tensorize_label(self, label):
        if type(label) == int:
            #tensor = [[0, 0, 0, 0]]
            #tensor[0][label] = 1
            tensor = torch.LongTensor([label])

            return tensor
