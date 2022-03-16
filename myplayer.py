import pynput.mouse    as ms
import pynput.keyboard as kb
from pynput.mouse import Button
import csv
import os.path
from os import path
from time import sleep
import cv2
from PIL import ImageGrab
import numpy as np
import pickle

from pynput.mouse import Listener

class Player:

    def __init__(self, piles):
        self.piles = piles

        self.real_board = ((0,),) * piles

        self.mouse = ms.Controller()
        self.is_settings_configured = False
        self.is_cards_configured = False
        self.coords = None
        self.buttons = ['hard_game','find_game', 'tutorial_x', 'card_pos', 'pile0', 'pile1', 'pile2', 'pile3', 'return_lobby']
        self.possible_cards = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']

        self.settings = {}
        self.cards = {}


        if path.exists("player_settings.csv"):
            with open('player_settings.csv') as csvfile:
                readcsv = csv.reader(csvfile, delimiter = ",")
                for row in readcsv:
                    self.settings[row[0]] = [int(row[1]), int(row[2])]
            self.is_settings_configured = True



        if path.exists("card_settings.pkl"):
            file_name = "card_settings.pkl"
            open_file = open(file_name, "rb")
            self.cards = pickle.load(open_file)
            open_file.close()
            self.is_cards_configured = True

    def click_pos(self, button_name):
        self.mouse.position = (self.settings[button_name][0], self.settings[button_name][1])
        self.mouse.press(Button.left)
        self.mouse.release(Button.left)

    def reset_real_game(self):
        self.click_pos('return_lobby')
        sleep(5)
        self.click_pos('hard_game')
        sleep(5)
        self.click_pos('find_game')
        sleep(5)
        self.click_pos('tutorial_x')
        sleep(1)
        self.real_board = ((0,),) * self.piles


    def get_card_img(self):
        d = 20
        x = 680 #self.settings['card_pos'][0]-d
        y = 890 #self.settings['card_pos'][1]-d

        dx = x+(d*2)
        dy = y+(d*2)

        img = ImageGrab.grab(bbox=(x, y, dx, dy))  # x, y, w, h
        #img.show()
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = 127
        img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


        return np.count_nonzero(img[1], axis=1)

    def setup(self):
        if input("Do you want to confiture Buttons? 1 or 0"):
            if self.is_settings_configured == False:
                for each in self.buttons:
                    if each == 'return_lobby':
                        input("Pausing untill return to lobby is pressable")
                        print("Now ready to read return to lobby button")
                    self.get_posistions(each)
                    self.settings[each] = self.coords
                    print(each, self.coords[0], )

                with open('player_settings.csv', 'w') as f:
                    for key in self.settings.keys():
                        f.write("%s,%s\n" % (key, f"{self.settings[key][0]},{self.settings[key][1]}"))

        if input("Do you want to confiture Cards? 1 or 0"):
            if [each for each in self.possible_cards if each not in self.cards] is not None:
                for i in range(13):
                    card_no = int(input("Card no"))
                    img = self.get_card_img()
                    self.cards[str(card_no)] = img
                    print([each for each in self.possible_cards if each not in self.cards])

                file_name = "card_settings.pkl"
                open_file = open(file_name, "wb")
                pickle.dump(self.cards, open_file)
                open_file.close()





    def on_click(self, x, y, button, pressed):
        print (f"Mouse clicked at {x} : {y}")
        self.coords = self.mouse.position
        if not pressed:
            #self.listener.stop()
            print("Listener stopped")
            return False


    def get_posistions(self, target):
        self.coords = None
        print(f"Hover mouse over {target} and click it.")
        with ms.Listener(on_click=self.on_click) as listener:
            listener.join()
        while self.coords == None:
            pass



    def draw_card(self):
        winner = -1
        error = 99
        sleep(1)
        while error > 90:
            actual_card = self.get_card_img()
            for key in self.cards.keys():
                test_error = np.sum(np.absolute((self.cards[key]- actual_card)))
                if error > test_error:
                    error = test_error
                    winner = key
                    

        winner = int(winner)
        if winner > 10:
            winner = 10

        print(f"I think the card is {winner}")

        return winner





    def get_real_action(self, action, board, card):
        value_of_pile = board.piles[action]
        print(f"Real board: {self.real_board}")

        for i in range(len(self.real_board)):
            #a = self.real_board[i]
            if value_of_pile == self.real_board[i]:
                winner = i
                break
        sleep(1)
        self.click_pos(f"pile{winner}")

        pile = list(self.real_board[winner])
        n_pile = []
        for elem in pile:
            n_pile.append(elem + card)
            if card == 1:
                n_pile.append(elem + card + 10)
                break

        if any(x == 21 for x in n_pile):
            n_pile = [0]
        if n_pile[0] > 21:
            is_terminal = True
        else:
            n_pile = [x for x in n_pile if x <= 21]
        n_piles = list(self.real_board)
        n_piles[winner] = tuple(n_pile)

        self.real_board = tuple(n_piles)

        return