import torch
import torch.optim as optim

from neural import Net
from collections import deque
import itertools




class Agent:
    def __init__(self, save_dir, checkpoint=None):
        self.save_dir = save_dir
        self.net = Net()
        self.loss_fn = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=0.00001, momentum=0.9)
        self.memory = deque(maxlen=1000)

        self.use_cuda = False
        self.save_every = 10

        if checkpoint:
            self.load(checkpoint)

    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""
        output = self.net.forward(state.tensorize())
        output.item()


        return output.item()

    def cache(self, experience):
        """Add the experience to memory"""
        """Experience: Board, score"""


        pass

    def recall(self):
        """Sample experiences from memory"""
        pass

    def learn(self, input, label):
        self.optimizer.zero_grad()
        output = self.net.forward(input.tensorize())
        label = torch.tensor([label])
        loss = self.loss_fn(output, label)
        loss.backward()
        self.optimizer.step()

        return loss.item(), output.item()

        """learn, given state and true result !!! for classification
        self.optimizer.zero_grad()
        output = self.net.forward(input.tensorize())
        predicted_card = torch.argmax(output).item()
        label = self.tensorize_label(label)
        output = output.unsqueeze(dim=0)
        loss = self.loss_fn(output, label)
        loss.backward()
        self.optimizer.step()

        return loss.item(), predicted_card
        """

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
