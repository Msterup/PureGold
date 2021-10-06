import numpy as np

# the follow two objects are used to compress the storage of flattens
possible_board_states = np.arange(23**6).reshape((23, 23, 23, 23, 23, 23))
# follow should not depend on game mode
possible_deck_states = np.arange(5**9).reshape((5, 5, 5, 5, 5, 5, 5, 5, 5))