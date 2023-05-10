#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:22:15 2023

@author: manuelestevez
"""
import argparse
import logging
import os
import pickle
import random
from collections import defaultdict
from typing import Optional, List

import numpy as np
from tqdm import tqdm

from topoworld.utils.vis_maze import draw_maze_collection

from topoworld.utils.maze import max_pi


def dfs(adj_list, visited, vertex, result, key):
    visited.add(vertex)
    result[key].append(vertex)
    for neighbor in adj_list[vertex]:
        if neighbor not in visited:
            dfs(adj_list, visited, neighbor, result, key)


def generate_random_maze_adjacency(lx: int, lz: int, pi: int = 0):
    nb = {}
    # neighbors dictionary
    for i in range(lx * lz):
        if i % lx == 0:
            if np.floor(i / lx) == 0:
                nb[i] = [i + 1, i + lx]
            elif np.floor(i / lx) == lz - 1:
                nb[i] = [i - lx, i + 1]
            else:
                nb[i] = [i - lx, i + 1, i + lx]
        elif i % lx == lx - 1:
            if np.floor(i / lx) == 0:
                nb[i] = [i - 1, i + lx]
            elif np.floor(i / lx) == lz - 1:
                nb[i] = [i - lx, i - 1]
            else:
                nb[i] = [i - lx, i - 1, i + lx]
        else:
            if np.floor(i / lx) == 0:
                nb[i] = [i - 1, i + 1, i + lx]
            elif np.floor(i / lx) == lz - 1:
                nb[i] = [i - lx, i - 1, i + 1]
            else:
                nb[i] = [i - lx, i - 1, i + 1, i + lx]

    A = set()  # initial configuration (snake)
    edges = set()  # all possible edges
    for i in range(lx * lz):
        for j in range(lx * lz):
            if j in nb[i]:
                edges.add(frozenset({i, j}))
                if np.floor(i / lx) % 2 == 0:
                    if j != i + 1:
                        if j % lx == lx - 1 and j > i:
                            A.add(frozenset({i, j}))
                        else:
                            continue
                    else:
                        A.add(frozenset({i, j}))
                else:
                    if j != i + 1:
                        if j % lx == 0 and j > i:
                            A.add(frozenset({i, j}))
                        else:
                            continue
                    else:
                        A.add(frozenset({i, j}))
            else:
                continue

    for _ in range(pi):
        A.add(random.sample(tuple(edges - A), 1)[0])

    if edges - A:
        repeat = 100000

        for i in range(repeat):
            NA = edges - A
            A1 = set()
            RA = random.choice(list(A))
            RNA = random.choice(list(NA))
            for r in A:
                A1.add(r)
            A1.remove(RA)
            A1.add(RNA)
            adj_list = defaultdict(list)
            for x, y in A1:
                adj_list[x].append(y)
                adj_list[y].append(x)
            result = defaultdict(list)
            visited = set()
            for vertex in adj_list:
                if vertex not in visited:
                    dfs(adj_list, visited, vertex, result, vertex)
            logging.debug(result.values())
            if len(result.values()) == 1 and len(visited) == lx * lz:
                A = A1
                logging.debug(("cnx"))
                logging.debug((len(result.values())))
            else:
                logging.debug(('discnx'))
                logging.debug((len(result.values())))

    # Adjacency matrix
    Ap = np.zeros((lx * lz, lx * lz))
    for i in nb:
        for j in nb:
            if frozenset({i, j}) in A:
                Ap[i, j] = 1
            else:
                continue

    return Ap


def sample_random_mazes(lx: int, lz: int, nm: int, save_pickle: bool = False, pis: Optional[List[int]] = None):
    """
    randomly generate collection of lx x lz mazes
    @param pis: list of pis to sample from
    @param max_pi: maximum fundamental cycle to sample from
    @param save_pickle: if True, write pickled object to disc
    @param lx: width of the maze
    @param lz: height of te maze
    @param nm: number of independently sampled mazes per pi
    @return: collection of mazes
    """
    run_name = f"mazes_{lx}x{lz}_{random.getrandbits(32):X}"

    os.makedirs("mazes", exist_ok=True)

    if pis is None:
        mp = max_pi(lx, lz)
        all_pis = list(range(mp + 1))
        pis = [all_pis[0], all_pis[mp//2], all_pis[mp]]
    else:
        pis = list(set(pis).intersection(set(range(max_pi(lx, lz) + 1))))

    pis.sort()

    print(f"Generate {args.lx}x{args.lz} mazes with pi_1={pis}")
    logging.info(f"Generate {args.lx}x{args.lz} mazes with pi_1={pis}")

    mazes = {"mazes": {pi: [generate_random_maze_adjacency(lx, lz, pi=pi) for _ in tqdm(range(nm))] for pi in pis},
             "info": {"lx": lx, "lz": lz, "n_mazes": nm, "pis": pis}}

    if save_pickle:
        path = os.path.join("mazes", f"{run_name}.p")
        print(f"Save at {path}")
        logging.info(f"Save at {path}")
        pickle.dump(mazes, open(path, "wb"))

    return mazes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Maze Sampler', description='Generate collection of rectangular mazes of a '
                                                                      'given size sorted by their fundamental group',
                                     epilog='Have fun torturing your agent ;)')

    parser.add_argument('-x', '--lx', help="maze width", type=int, default=4)
    parser.add_argument('-z', '--lz', help="maze height", type=int, default=4)
    parser.add_argument('-n', '--n_envs', help="sample n mdp per pi", type=int, default=5)
    parser.add_argument('--pis', nargs="+", type=int, default=None)

    args = parser.parse_args()

    mazes = sample_random_mazes(args.lx, args.lz, args.n_envs, pis=args.pis, save_pickle=True)

    draw_maze_collection(mazes)
