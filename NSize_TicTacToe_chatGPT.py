#!/usr/bin/python3
from typing import List, Optional, Tuple
from typing import List, Union
from io import TextIOWrapper, BufferedWriter
import numpy as np
import sys
import argparse
import random as r
from functools import cache
from time import time
from tqdm import tqdm

boardSize = 4


class TicTacToeBoard:
    def __init__(self, board: List[List[int]] | None = None, size=3, turn=1, mode=0):
        if board is None:
            self.board: List[List[int]] = []
            for i in range(size):
                self.board.append([0] * size)
        else:
            self.board = board
        self.turn = turn
        self.size = size
        self.mode = mode

    def make_move(self, x: int, y: int):
        new_board = [row[:] for row in self.board]
        new_board[x][y] = self.turn
        return TicTacToeBoard(new_board, self.size, 1 if self.turn == 2 else 2, self.mode)

    def printToFile(self, file: TextIOWrapper | BufferedWriter):
        if self.mode == 0 and type(file) == TextIOWrapper:
            file.write(str(self) + '\n')
        elif self.mode == 1 and type(file) == BufferedWriter:
            size = int((self.size ** 2) / 8) + 1
            bitsetX = np.zeros(size, dtype=np.uint8)
            bitsetO = np.zeros(size, dtype=np.uint8)
            for i in range(self.size):
                for j in range(self.size):
                    if self.board[i][j] == 1:
                        bitsetX[(i * self.size + j) //
                                8] |= 1 << (i * self.size + j) % 8
                    elif self.board[i][j] == 2:
                        bitsetO[(i * self.size + j) //
                                8] |= 1 << (i * self.size + j) % 8

            file.write(bytes([self.turn - 1]))
            file.write(bitsetX.tobytes())
            file.write(bitsetO.tobytes())

    def __str__(self):
        s = f'{"X" if self.turn == 1 else "O"}'
        for row in self.board:
            for cell in row:
                s += f'{"X" if cell == 1 else "O" if cell == 2 else " "}'
        return s

    def getChildren(self) -> List['TicTacToeBoard']:
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    moves.append(self.make_move(i, j))
        return moves

    @cache
    def isFinal(self) -> bool:
        # test if there is a winner
        for i in range(self.size):
            for j in range(self.size - 2):
                if self.board[i][0+j] == self.board[i][1+j] == self.board[i][2+j] != 0:
                    return True
                if self.board[0+j][i] == self.board[1+j][i] == self.board[2+j][i] != 0:
                    return True

        v1 = self.board[0][0]
        v1Valid = v1 != 0
        v2 = self.board[0][self.size - 1]
        v2Valid = v2 != 0

        for i in range(1, self.size):
            if v1 != 0 and self.board[i][i] != v1:
                v1Valid = False

            if v2 != 0 and self.board[i][self.size - 1 - i] != v2:
                v2Valid = False

        if v1Valid or v2Valid:
            return True

        # test if board is full
        found_zero = False
        for row in self.board:
            for cell in row:
                if cell == 0:
                    found_zero = True
                    break
            if found_zero:
                break
        if not found_zero:
            return True

        return False

    def __eq__(self, other: 'TicTacToeBoard'):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(str(self))


def minimax(board: TicTacToeBoard, depth: int, alpha: float, beta: float, maximizingPlayer: bool) -> Tuple[int, Optional[Tuple[int, int]]]:
    if depth == 0 or board.isFinal():
        return evaluate(board), None

    if maximizingPlayer:
        maxEval = float('-inf')
        bestMove = None
        for i, row in enumerate(board.board):
            for j, cell in enumerate(row):
                if cell == 0:  # Check if the cell is empty, indicating a possible move
                    child = board.make_move(i, j)
                    eval, _ = minimax(child, depth - 1, alpha, beta, False)
                    if eval > maxEval:
                        maxEval = eval
                        bestMove = (i, j)  # Store the coordinates of the move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            if beta <= alpha:
                break
        return maxEval, bestMove
    else:
        minEval = float('inf')
        bestMove = None
        for i, row in enumerate(board.board):
            for j, cell in enumerate(row):
                if cell == 0:  # Check if the cell is empty, indicating a possible move
                    child = board.make_move(i, j)
                    eval, _ = minimax(child, depth - 1, alpha, beta, True)
                    if eval < minEval:
                        minEval = eval
                        bestMove = (i, j)  # Store the coordinates of the move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            if beta <= alpha:
                break
        return minEval, bestMove


def evaluate(board: TicTacToeBoard) -> int:
    # This is a placeholder evaluation function.
    # You need to replace it with the actual logic to evaluate the board.
    # For Tic-Tac-Toe, it can return a positive value if the maximizing player is winning,
    # a negative value if the minimizing player is winning, or 0 for a draw.
    # For simplicity, let's say +1 for player 1 win, -1 for player 2 win, 0 otherwise.
    if board.isFinal():
        if board.turn == 1:
            return boardSize ** 2 - 1
        else:
            return -boardSize ** 2 + 1
        pass
    return 0

# Example of calling the minimax function with alpha-beta pruning
# initial_board = TicTacToeBoard()
# best_score, best_move = minimax(initial_board, depth=9, alpha=float('-inf'), beta=float('inf'), maximizingPlayer=True)
# print(f"Best score: {best_score}, Best move: {best_move}")


boards = []
with open("dataset4x4.txt", "r") as f:
    size = int(f.readline())
    for i in range(100_000):
        board = TicTacToeBoard()
        boardstr = f.readline()
        playing = 1 if boardstr[0] == "X" else 2
        boardstr = boardstr[1:]
        board.board = [[1 if boardstr[j * size + i] ==
                        "X" else 2 if boardstr[j * size + i] == "O" else 0
                        for i in range(size)] for j in range(size)]

        board.turn = playing

        boards.append(board)

s = 0
for board in tqdm(boards):
    start = time()
    best_score, best_move = minimax(board, depth=9, alpha=float(
        '-inf'), beta=float('inf'), maximizingPlayer=True)
    # print(f"Best score: {best_score}, Best move: {best_move}")
    end = time()

    s += end - start

print(s)
