import torch
import torch.optim as optim

from neural import ResNet, ResidualBlock
from game import YukonBoard

from collections import deque
import random as rn
import itertools
from functools import lru_cache
import torch_xla
import torch_xla.core.xla_model as xm



class RegAgent:
    def __init__(self, save_dir, checkpoint=None):
        self.save_dir = save_dir



        # Acquires the default Cloud TPU core and moves the model to it
        self.device = xm.xla_device()
        self.net = ResNet(ResidualBlock, 5).to(self.device)

        if checkpoint:
            self.load(checkpoint)

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.0002, betas=(0.09, 0.0999), eps=1e-08, weight_decay=0.0005, amsgrad=False)
        #self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00225)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=0.00001, momentum=0.9)

        self.LSTM_size =     12000
        self.recall_size =   100
        self.recall_chance = 1
        self.recall_min =   self.recall_size*self.recall_chance

        self.memory = [] # deque(maxlen=self.LSTM_size)

        self.use_cuda = True
        self.save_every = 10
        self.prediction_rate = 0

        self.num_cached = 0

        self.nik_rate = 1.10



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
                output = self.net.forward(each.tensorize().to(self.device))
                act_total += output.item()

            act_mean = act_total/len(act_list)

            return act_mean

        else:
            return self.net.forward(board.tensorize().to(self.device)).item()


    def cache(self, board, score):
        permute = False

        if permute:
            """Add the experience to memory"""
            """Experience: Board, score"""
            permutations = itertools.permutations(board.piles)
            unique_permutations = []
            for each in permutations:
                unique_permutations.append(each)
            unique_permutations = set(unique_permutations)

            for each in unique_permutations:
                self.memory.append((YukonBoard(each, deck=board.deck, card=board.card, turn=True, terminal=False), score))

        else:
            self.memory.append((board, score))

        return

    def recall(self):
        """Sample experiences from memory"""
        num_cached = self.num_cached
        self.num_cached = 0
        return rn.sample(self.memory, num_cached), num_cached


    def learn(self):
        self.net.train(True)
        sum_loss = 0
        rn.shuffle(self.memory)
        for input, label in self.memory:
            self.optimizer.zero_grad()
            output = self.net.forward(input.tensorize().to(self.device))
            label = torch.tensor([label])
            loss = self.loss_fn(output, label)
            loss.backward()
            xm.optimizer_step(optimizer)
            xm.mark_step()
            sum_loss += loss.item()

        print(f"Number of cached experiences: {len(self.memory)}")
        num_experiences = len(self.memory)
        num_delete = int(num_experiences*0.2)
        self.memory = self.memory[:num_delete]
        return sum_loss/num_experiences

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



    def save(self, e):
        """Saves the nural net in a  """
        save_path = self.save_dir / f"mario_net_{int(e // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                nik_rate = self.nik_rate
            ),
            save_path
        )
        print(f"NeuralNet saved to {save_path} at step {e}")

        pass

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=(self.device))
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
