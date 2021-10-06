from cpython cimport array
import array


import numpy as np
import random as rn
from copy import deepcopy
import time



cpdef int nikolai(board):
    cdef int size = len(board.piles)
    cdef int card = board.card
    cdef int num10 = board.deck[9]
    cdef int valg = -1


    table = []
    aces = []
    for each in board.piles:
        table.append(min(each))
        if len(each) == 1:
            aces.append(0)
        else:
            aces.append(1)
    card = board.card

    deck = []
    prob = np.zeros(10)
    for i in range(10):
        prob[i] = board.deck[i] / 4
        #for j in range(board.deck[i]):
            #deck.append(i+1)

    aces = np.array(aces)
    table = np.array(table)
    #deck = np.array(deck)
    #np.random.shuffle(deck)

    # start af valg
    a = np.where(table + card == 21)
    if np.size(a) > 0:
        valg = a[0][0]
        return valg


    a = np.where(table + card == 11)

    if np.size(a) > 0 and (num10 > 0 or aces[a[0][0]] == 1 or card == 1):
        valg = a[0][0]
        return valg


    a = np.where(table + card == 20)
    if np.size(a) > 0 and card == 10:
        valg = a[0][0]
        return valg


    if card == 10:  # hvis sansynligheden for at lave 21 stiger, gør
        #prob = np.zeros(9)
        #for h in range(1, 10):
            #prob[h - 1] = sum(deck == h) / 4
        valg = -1
        p = 0
        for h in range(1, 10):
            Table = deepcopy(table)
            Table[(Table == 11) | (Table == 1) | Table == 10] = -99
            for test in range(size):
                if sum(Table == Table[test] + card) > 0:
                    Table[test] = -99

            a = np.where(Table + card + h == 21)
            before = 0
            if np.size(a) > 0 and Table[a[0][0]] >= 12:
                before = prob[21 - Table[a[0][0]] - 1]  # nok rigitg
            if np.size(a) > 0 and prob[h - 1] - before > p:
                valg = a[0][0]
                a = prob[h - 1] - before

        if valg > -1:
            return valg


        a = np.where(table + card == 10)
        if np.size(a) > 0:
            valg = a[0][0]
            return valg


    a = np.where(table + card == card)
    if np.size(a) > 0 and sum(table == 0) - sum(table == card) >= 1:
        valg = a[0][0]
        return valg


    prob = np.zeros(10)
    for i in range(10):
        prob[i] = board.deck[i] / 4
    valg = -1
    p = 0
    for h in range(1, 10):
        Table = deepcopy(table)
        Table[(Table == 11) | (Table == 1) | (Table + card == 10)] = -99
        for test in range(size):
            if sum(Table == Table[test] + card) > 0:
                Table[test] = -99

        a = np.where(Table + card + h == 21)
        before = 0
        if np.size(a) > 0 and Table[a[0][0]] >= 12:
            before = prob[21 - Table[a[0][0]] - 1]  # nok rigitg
        if np.size(a) > 0 and prob[h - 1] - before > p:
            valg = a[0][0]
            a = prob[h - 1] - before
    if valg > -1:
        return valg


    prob = np.zeros(10)
    for i in range(10):
        prob[i] = board.deck[i] / 4
    valg = -1
    p = 0
    for h in range(1, 10):
        Table = deepcopy(table)
        Table[(Table == 11) | (Table == 1)] = -99
        for test in range(size):
            if sum(Table == Table[test] + card) > 0:
                Table[test] = -99

        a = np.where(Table + card + h == 11)
        before = 0
        if np.size(a) > 0 and Table[a[0][0]] >= 12:
            before = prob[11 - Table[a[0][0]] - 1]  # nok rigitg
        if np.size(a) > 0 and prob[h - 1] - before > p:
            valg = a[0][0]
            a = prob[h - 1] - before
    if valg > -1:
        return valg


    for yo in [2, 3, 4, 6, 7, 8, 9]:
        if sum(table == yo) >= 2:
            a = np.where(table == yo)
    if np.size(a) > 0:
        valg = a[0][0]
        return valg


    a = np.where((table + card < 21) & (table + card < 10) & (table + card != card) & (table != 1) & (table != 11))
    if np.size(a) > 0:
        valg = a[0][0]
        return valg


    a = np.where((table + card < 21) & (table + card < 10) & (table + card != card) & (table != 10) & (table != 11))
    if np.size(a) > 0:
        valg = a[0][0]
        return valg


    a = np.where((table + card < 21) & (table != 1) & (table != 11))
    if np.size(a) > 0:
        valg = a[0][0]
        return valg


    # final choice random
    a = np.where(table + card < 21)
    if np.size(a) > 0:
        valg = a[0][0]
        return valg


    return 0