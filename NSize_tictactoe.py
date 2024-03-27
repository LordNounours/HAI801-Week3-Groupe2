"""
Tic Tac Toe Player
"""

import math
import numpy as np

X = "X"
O = "O"
EMPTY = None
SIZE = 4


def initial_state():
    """
    Returns starting state of the board.
    """
    board = [[EMPTY] * SIZE for _ in range(SIZE)]
    return board



def player(board):
    """
    Returns player who has the next turn on a board.
    """
    cnt_X = 0
    cnt_o = 0
    # Count X and O's on the game board
    for i in range(SIZE):
        for j in range(SIZE): 
            if board[i][j] == X:
                cnt_X += 1
            elif board[i][j] == O:
                cnt_o += 1

    if cnt_X == cnt_o:
        return X
    elif cnt_X > cnt_o:
        return O
    else:
        return X



def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # Each action is represented as a tuple (i, j) where i corresponds to the row of the move (0, 1, or 2) and j corresponds to which cell in the row corresponds to the move (also 0, 1, or 2).
    actions = set()
    for i in range(SIZE):
        for j in range(SIZE):
            if board[i][j] == None:
                actions.add((i, j))
    
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # If action is not a valid action for the board, the program should raise an exception
    if action not in actions(board):
        raise Exception("Invalid Action")
    
    #  Make a copy of the board (Important, else can run into some hard to debug state errors)
    new_board = [row[:] for row in board]

    # making move (i, j) on the board
    new_board[action[0]][action[1]] = player(board)

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Return tictactoe winner if there is one
    # Check rows
    for row in board:
        if all(cell == X for cell in row):
            return X
        elif all(cell == O for cell in row):
            return O

    # Check columns
    for col in range(len(board[0])):
        if all(board[row][col] == X for row in range(len(board))):
            return X
        elif all(board[row][col] == O for row in range(len(board))):
            return O

    # Check diagonals
    if all(board[i][i] == X for i in range(len(board))):
        return X
    elif all(board[i][i] == O for i in range(len(board))):
        return O
    if all(board[i][len(board)-1-i] == X for i in range(len(board))):
        return X
    elif all(board[i][len(board)-1-i] == O for i in range(len(board))):
        return O

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Check if there is a winner
    if winner(board) != None:
        return True
    else:
        # Check if the board is full
        for i in range(SIZE):
            for j in range(SIZE):
                if board[i][j] == None:
                    return False
        return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def max_value(board, alpha, beta, ite):
    """
    Returns the maximum value for the current player on the board 
    using alpha-beta pruning.
    """
    # If the depth limit is reached, return the utility of the board
    if ite > 6:
        return utility(board)
    if terminal(board):
        return utility(board)
    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action), alpha, beta, ite+1))
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    return v

def min_value(board, alpha, beta, ite):
    """
    Returns the minimum value for the current player on the board 
    using alpha-beta pruning.
    """
    # If the depth limit is reached, return the utility of the board
    if ite > 6:
        return utility(board)
    if terminal(board):
        return utility(board)
    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action), alpha, beta, ite+1))
        beta = min(beta, v)
        if alpha >= beta:
            break
    return v

def minimax(board):
    """
    Returns the optimal action for the current player on the board 
    using the minimax algorithm with alpha-beta pruning.
    """
    if terminal(board):
        return None

    if player(board) == X:
        v = -math.inf
        opt_action = None
        for action in actions(board):
            new_value = min_value(result(board, action), -math.inf, math.inf, ite=0)
            if new_value > v:
                v = new_value
                opt_action = action
        return opt_action
    
    elif player(board) == O:
        v = math.inf
        opt_action = None
        for action in actions(board):
            new_value = max_value(result(board, action), -math.inf, math.inf, ite=0)
            if new_value < v:
                v = new_value
                opt_action = action
        return opt_action

    
if __name__ == "__main__":
    board = initial_state()
    # Read the file of all possible board states
    file_path = "4.txt"
    try:
      file = open(file_path, "r")
    except IOError:
      print("Error: File does not appear to exist.")
      exit()
    lines = file.readlines()
    # For each board state, run the minimax algorithm
    for line in lines:
        k = 1
        for i in range(SIZE):
            for j in range(SIZE):
                if line[k] == " " : board[i][j] = None
                else : board[i][j] = line[k]
                k += 1
        print("==> Initial Board")
        for grid in board:
            print(grid)
        #Minmax
        while not terminal(board):
            action = minimax(board)
            board = result(board, action)
        #Print the final board
        print("==> Final Board")
        for grid in board:
            print(grid)
        print("Winner: ", winner(board))
        print(" ----------------- ")