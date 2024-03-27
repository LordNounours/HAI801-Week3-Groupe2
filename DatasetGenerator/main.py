#!/usr/bin/python3

from typing import List, Union
from io import TextIOWrapper, BufferedWriter
import numpy as np
import sys
import argparse
import random as r
from functools import cache
from time import time


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


s = set()

parser = argparse.ArgumentParser(description='Generate dataset for TicTacToe')
parser.add_argument('--size', "-s", type=int, required=True,
                    help='Size of the board')
parser.add_argument('--output', "-o", type=str,
                    required=True, help='Output file')
parser.add_argument('--samples', "-n", type=int, default=100000,
                    help='Number of samples to generate')
parser.add_argument('--mode', "-m", type=str, default='ascii', choices=['ascii', 'bitset'],
                    help='Output mode, ascii or bitset')
parser.add_argument('--seed', "-r", type=int, default=time(),
                    help='Random seed (default: current time)')

args = parser.parse_args()

r.seed(args.seed)

k = 0


def generateDataset(board: TicTacToeBoard, file: TextIOWrapper | BufferedWriter, depth=0):
    global k
    if board.isFinal():
        return

    # print(board)
    board.printToFile(file)
    l = board.getChildren()
    r.shuffle(l)
    k += 1
    for child in l:
        if (child.isFinal() or k >= args.samples):
            return

        if (k % 100 == 0):
            bar_length = 50
            progress = k / args.samples

            print("[" + "=" * (int(bar_length * progress) - 1) + ">" + " " * (bar_length -
                  int(bar_length * progress)) + f"] {int(progress * 100)}%   ", end="\r")

        h = hash(child)
        if h not in s:
            s.add(h)
            generateDataset(child, file, depth + 1)


if args.mode == 'ascii':
    with open(args.output, 'w') as file:
        file.write(f'{args.size}\n')
        generateDataset(TicTacToeBoard(size=args.size, mode=0), file)
        print("[" + "=" * 50 + "]" + " 100%   ", end="\n")
        # for item in s:
        #     item.printToFile(file)

else:
    with open(args.output, 'wb') as file:
        file.write(bytes([args.size]))
        generateDataset(TicTacToeBoard(size=args.size, mode=1), file)
        print("[" + "=" * 50 + "]" + " 100%   ", end="\n")
        # for item in s:
        #     item.printToFile(file)
