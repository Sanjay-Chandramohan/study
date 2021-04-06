from random import randrange
from  random import seed
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import sys
from trueskill import Rating, quality_1vs1, rate_1vs1
import time
import itertools
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

class Hexboard:
    BLUE = 1
    RED = 2
    EMPTY = 3
    def __init__(self, board_size):
        self.board = {}
        self.size = board_size
        self.game_over = False
        self.turn = Hexboard.BLUE
        self.moves = []
        for x in range(board_size):
            for y in range (board_size):
                self.board[x,y] = Hexboard.EMPTY

    def get_board(self):
        return self.board

    def get_board_size(self):
        return self.size

    def get_moves(self):
        return self.moves

    def get_turn(self):
        return self.turn

    def is_game_over(self):
        return self.game_over

    def get_key(self):
        key = ''
        for x in range(self.size):
            for y in range(self.size):
                key += str(self.board[(x,y)])
        return key

    def get_result(self):
        if self.check_win(Hexboard.BLUE): #p1 wins
            return Hexboard.BLUE
        elif self.check_win(Hexboard.RED): #p2 wins
            return Hexboard.RED
        return 0 #draw

    def is_empty(self, coordinates):
        return self.board[coordinates] == Hexboard.EMPTY

    def is_color(self, coordinates, color):
        return self.board[coordinates] == color

    def get_color(self, coordinates):
        if coordinates == (-1, -1):
            print("don't go here")
            return Hexboard.EMPTY
        return self.board[coordinates]

    def place(self, coordinates, color):
        if not self.game_over and self.board[coordinates] == Hexboard.EMPTY:
            self.board[coordinates] = color
            self.moves.append(coordinates)
            if self.check_win(Hexboard.RED) or self.check_win(Hexboard.BLUE) or self.is_board_full():
                self.game_over = True
            self.turn = self.get_opposite_color(self.turn)
            return True
        return False #invalid move

    def get_opposite_color(self, current_color):
        if current_color == Hexboard.BLUE:
            return Hexboard.RED
        return Hexboard.BLUE

    def get_neighbors(self, coordinates):
        (cx,cy) = coordinates
        neighbors = []
        if cx-1>=0:   neighbors.append((cx-1,cy))
        if cx+1<self.size: neighbors.append((cx+1,cy))
        if cx-1>=0    and cy+1<=self.size-1: neighbors.append((cx-1,cy+1))
        if cx+1<self.size  and cy-1>=0: neighbors.append((cx+1,cy-1))
        if cy+1<self.size: neighbors.append((cx,cy+1))
        if cy-1>=0:   neighbors.append((cx,cy-1))
        return neighbors

    def border(self, color, move):
        (nx, ny) = move
        return (color == Hexboard.BLUE and nx == self.size-1) or (color == Hexboard.RED and ny == self.size-1)

    def traverse(self, color, move, visited):
        if not self.is_color(move, color) or (move in visited and visited[move]): return False
        if self.border(color, move): return True
        visited[move] = True
        for n in self.get_neighbors(move):
            if self.traverse(color, n, visited): return True
        return False

    def check_win(self, color):
        for i in range(self.size):
            if color == Hexboard.BLUE: move = (0,i)
            else: move = (i,0)
            if self.traverse(color, move, {}):
                return True
        return False

    def print(self):
        print("   ",end="")
        for y in range(self.size):
            print(chr(y+ord('a')),"",end="")
        print("")
        print(" -----------------------")
        for y in range(self.size):
            print(y, "|",end="")
            for z in range(y):
                print(" ", end="")
            for x in range(self.size):
                piece = self.board[x,y]
                if piece == Hexboard.BLUE: print("b ",end="")
                elif piece == Hexboard.RED: print("r ",end="")
                else:
                    if x==self.size:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")
        print("   -----------------------")

    def is_board_full(self):
        #check if valid spaces exist
        full_board = True
        for x in range(self.size):
            for y in range(self.size):
                if self.is_empty((x,y)):
                    full_board = False
        if full_board:
            self.game_over = True
            return True
        return False

    def undo(self):
        coordinates = self.moves.pop()
        if coordinates in self.board:
            self.board[coordinates] = Hexboard.EMPTY
            self.turn = self.get_opposite_color(self.turn)
            self.game_over = False # Be sure to set this back if it was a winning move
            return True
        return False

# player class that encompasses all types of players
class Player:
    def __init__(self, num, type='r', eval='r', budget=3, c_param=1, id='n'):
        self.num = num # Player number
        self.type = type # Type of player. Human, random, monte carlo or alpha-beta.
        self.budget = budget # Both in terms of monte carlo simulations, as well as time for iterative deepening
        self.id = id # Iterative deepening: yes or no?
        self.eval = eval # Evaluation: random or Dijkstra
        self.c_param = c_param # C parameter for exploration vs. exploitation in monte carlo
        self.rating = Rating() # ELO rating. Not Electric Light Orchestra, though.

    # Getters and setters
    def get_num(self):
        return self.num

    def set_num(self, n):
        self.num = n

    def get_type(self):
        return self.type

    def get_budget(self):
        return self.budget

    def get_id(self):
        return self.id

    def get_eval(self):
        return self.eval

    def get_c_param(self):
        return self.c_param

    def get_rating(self):
        return self.rating

    def set_rating(self, r):
        self.rating = r


# Node class for building a game tree
class Node:
    def __init__(self, game, parent):
        self.game = game
        self.children = []
        self.parent = parent
        self.visits = 0
        self.wins = 0

    # Check if child is there, if it is, return
    def get_child(self, state):
        for c in self.children:
            if c.game.board == state:
                return c

        return False

    # Add a child to the current state
    def add_child(self, game):
        child = Node(game, self)
        if not self.get_child(game.board):
            self.children.append(child)
            return child
        return False

    # Backpropagation for MCTS
    def update_stats(self, wins):
        self.visits += 1
        self.wins += wins


# Human move. Takes input from human, checks if it is correct or not.
def human_move(game):
    while True: # Check for correct input
        pre_split = input("Input a coordinate in the form 'x y' to place a tile: ")
        coord_input = pre_split.split(' ')
        if len(coord_input) != 2:
            print("Wrong input.")
            continue
        if coord_input[0].isalpha() and coord_input[1].isdigit():
            coord_input[0] = ord(coord_input[0].upper()) - ord('A') # Convert to
            coord_input[1] = int(coord_input[1])
            if coord_input[0] not in range(game.size) or coord_input[1] not in range(game.size):
                print("Coordinates not in range of board.")
                continue
            coords = (coord_input[0], coord_input[1])
            print(coords)
            if game.place(coords, game.get_turn()):
                break
            print("Could not place a tile there, try again")
        else:
            print("Wrong input.")

# Random move
def random_move(game):
	while True:
		move = (randrange(game.size), randrange(game.size))
		if game.place(move, game.get_turn()):
			break
	return move

# Random evaluation function for Alpha-Beta
def random_eval(game):
    return randrange(0, game.size) #(game.size*game.size)/2)

# Returns the node with the smallest distance
def smallest_distance(queue, distances):
    min_distance = np.Inf
    best = (-1, -1)
    for n in queue:
        if distances[n] < min_distance:
            min_distance = distances[n]
            best = n
    if best == (-1, -1):
        print("WAIT!")
    return best

# Dijkstra evaluation function for Alpha-Beta
def dijkstra_eval(game, color):
    shortest_total_distance = game.size * game.size

    visited = set() # Keep track if node was already visited
    distances = {} # Distances for each node
    queue = []
    queue.append("B") # A beggining node, not existing on the board but useful for Dijkstra
    for x in range(game.size):
        for y in range (game.size):
            distances[(x,y)] = np.Inf

    distances["B"] = 0 # Set the distance for the current edgehex to zero

    while not len(queue) == 0:
        current = smallest_distance(queue, distances)

        if current != "B":
            neighbours = game.get_neighbors(current)
        else: # Beginning node
            if color == Hexboard.BLUE:
                neighbours = [(0,x) for x in range(game.size)]
            else:
                neighbours = [(x,0) for x in range(game.size)]

        for n in neighbours:
            if game.is_color(n, color): #is you
                dist_to_n = 0
                if n not in visited and n not in queue:
                    queue.append(n)
            elif game.is_color(n, game.get_opposite_color(color)): # is opponent
                continue
            elif game.is_empty(n):
                dist_to_n = 1
                if n not in visited and n not in queue:
                    queue.append(n)

            temp = distances[current] + dist_to_n
            if temp < distances[n]:
                distances[n] = temp
        queue.remove(current)
        visited.add(current)

    #check final hexes for smallest value
    if color == Hexboard.BLUE:
        endhexes = [(game.size-1,x) for x in range(game.size)]
    else:
        endhexes = [(x,game.size-1) for x in range(game.size)]
    for h in endhexes:
        if distances[h] < shortest_total_distance:
            shortest_total_distance = distances[h]
    return shortest_total_distance


# Minimax enhanced with alphabeta
def minmax(game, alpha, beta, maximize, depth, eval):
    visited = 1
    cutoff = 0
    if depth == 0 or game.is_game_over():
        if eval == "d":
            if maximize: #opponents turn
                m = dijkstra_eval(game, game.get_opposite_color(game.get_turn()))
            else: #players turn
                m = dijkstra_eval(game, game.get_turn())
        else:
            m = random_eval(game)
        return (m, -1, -1, visited, cutoff)

    if maximize: # Maximizing player
        best_m = -np.Inf
        current_m = -np.Inf
        best_x, best_y = -1, -1
        for x in range(game.size):
            for y in range(game.size):
                if game.place((x, y), game.get_turn()):
                    r = minmax(copy.deepcopy(game), alpha, beta, False, depth-1, eval)
                    visited += r[3]
                    cutoff += r[4]
                    current_m = max(current_m, r[0])
                    alpha = max(alpha, current_m)
                    if current_m > best_m:
                        best_x, best_y = x, y
                        best_m = current_m
                    if alpha >= beta:
                        cutoff += 1
                        return (best_m, best_x, best_y, visited, cutoff) # Cutoff

                    game.undo()
        return (best_m, best_x, best_y, visited, cutoff)

    else: # Minimizing player
        best_m = np.Inf
        current_m = np.Inf
        best_x, best_y = -1, -1
        for x in range(game.size):
            for y in range(game.size):
                if game.place((x, y), game.get_turn()):
                    r = minmax(copy.deepcopy(game), alpha, beta, True, depth-1, eval)
                    visited += r[3]
                    cutoff += r[4]
                    current_m = min(current_m, r[0])
                    beta = min(beta, current_m)
                    if current_m < best_m:
                        best_x, best_y = x, y
                        best_m = current_m
                    if beta <= alpha:
                        cutoff += 1
                        return (best_m, best_x, best_y, visited, cutoff) # Cutoff

                    game.undo()
        return (best_m, best_x, best_y, visited, cutoff)

# Minimax enhanced with alphabeta
def minmaxtt(game, alpha, beta, maximize, depth, ttable):
    visited = 0 # First, visited is 0 because we're looking in the transposition table
    cutoff = 0
    score = ttable.get(game.get_key())
    if score != None: # Hit in transposition table
        return score + (visited,cutoff)

    visited = 1 # Game was not found in transposition table, so we visit a node

    if game.is_game_over() or depth <= 0:
        if maximize: #opponents turn
            m = dijkstra_eval(game, game.get_opposite_color(game.get_turn()))
        else: #players turn
            m = dijkstra_eval(game, game.get_turn())
        return (m, -1, -1, visited, cutoff)

    if maximize: # Maximizing player
        best_m = -np.Inf
        current_m = -np.Inf
        best_x, best_y = -1, -1
        for x in range(game.size):
            for y in range(game.size):
                if game.place((x, y), game.get_turn()):
                    r = minmaxtt(copy.deepcopy(game), alpha, beta, False, depth-1, ttable)
                    visited += r[3]
                    cutoff += r[4]
                    current_m = max(current_m,r[0])
                    alpha = max(alpha, current_m)
                    ttable[game.get_key()] = (current_m,x,y)
                    if current_m > best_m:
                        best_x, best_y = x, y
                        best_m = current_m
                    if alpha >= beta:
                        cutoff += 1
                        return (best_m, best_x, best_y, visited, cutoff) # Cutoff
                    game.undo()
        return (best_m, best_x, best_y, cutoff)

    else: # Minimizing player
        best_m = np.Inf
        current_m = np.Inf
        best_x, best_y = -1, -1
        for x in range(game.size):
            for y in range(game.size):
                if game.place((x, y), game.get_turn()):
                    r = minmaxtt(copy.deepcopy(game), alpha, beta, True, depth-1, ttable)
                    visited += r[3]
                    cutoff += r[4]
                    current_m = min(current_m, r[0])
                    beta = min(beta, current_m)
                    ttable[game.get_key()] = (current_m,x,y)
                    if current_m < best_m:
                        best_x, best_y = x, y
                        best_m = current_m
                    if beta <= alpha:
                        cutoff += 1
                        return (best_m, best_x, best_y, visited, cutoff) # Cutoff
                    game.undo()
        return (best_m, best_x, best_y, visited, cutoff)

# Calls minimax, either with or without iterative deepening and transposition tables
def iterative_deepening(game, p):
    f = ()
    if p.get_id() == 'n': # No iterative deepening, so no transposition tables either
        temp = minmax(game, -np.Inf, np.Inf, False, p.get_budget(), p.get_eval())
        if temp[1] != -1 and temp[2] != -1: # Only update if coordinates are valid
            f = temp
    else: # Iterative deepening and transposition tables
        d = 1
        ttable = {}
        starttime = time.time()
        while time.time()-starttime <= p.get_budget() and d <= game.get_board_size()*game.get_board_size()-len(game.get_moves()): #limit search time
            temp = minmaxtt(game, -np.Inf, np.Inf, False, copy.deepcopy(d), ttable)
            if temp[1] != -1 and temp[2] != -1: # Only update if coordinates are valid
                f = temp
            d += 1 # Increase depth
    return f

# Monte Carlo Tree Search
def fully_expanded(node):
    for i in range(node.game.size):
        for j in range(node.game.size):
            game = copy.deepcopy(node.game)
            if game.place((i,j), game.get_turn()): # Place all possible moves
                if not node.get_child(game.board): # If a new move is found
                    return False # Return: node is not fully expanded, and return the move to be expanded
    return True # This node was fully expanded


def expand(node): # Add a node to the game tree
    if node.game.is_game_over(): # If the game is over (there are no children)
        return None

    while True: # Try random moves
        i = randrange(node.game.size)
        j = randrange(node.game.size)
        game = copy.deepcopy(node.game)
        if game.place((i,j), game.get_turn()): # Place this random move
            if not node.get_child(game.board): # If a new move is found
                return node.add_child(game) # Add child to tree (expand) and return

# Select a node
def select(node, c_param = 1.0):
    while not node.game.is_game_over() and fully_expanded(node):
        node = best_child(node, c_param) # Traverse down path of the best
    return expand(node) or node

# Return the child with the highest UCT value
def best_child(node, c_param = 1.0):
    choices_weights = [(c.wins / c.visits) + c_param * np.sqrt((np.log(node.visits)/c.visits))
                        for c in node.children
    ] # Calculate UCT value for all nodes
    return node.children[np.argmax(choices_weights)] # Return node with largest value

# Get the child with the most visits
def winning_child(node):
    visits = [c.visits for c in node.children]
    return node.children[np.argmax(visits)] # Return most visited node

# Simulate the game with random playouts
def rollout(node, root_color):
    game = copy.deepcopy(node.game)
    opposite_player = game.get_turn() # there's 1 tile placed so its the opposite players turn
    while not game.is_game_over():
        random_move(game)

    if game.get_result() == root_color: #game.get_opposite_color(opposite_player): # We win
        return 1
    elif game.get_result() == node.game.get_opposite_color(root_color): # We lose
        return -1
    else: # It's a draw
        return 0

# Backpropagate the score to the nodes
def backpropagate(node, simulation_result):
    node.update_stats(simulation_result)
    if node.parent != None:
        backpropagate(node.parent, simulation_result)

# Monte Carlo Tree Search (MCTS) main loop
def monte_carlo_tree_search(root, player):
    i = 0
    while i < player.get_budget():
        leaf = select(root, player.get_c_param())
        simulation_result = rollout(leaf, root.game.get_turn())
        backpropagate(leaf, simulation_result)
        i += 1
    return copy.deepcopy(winning_child(root).game)

# Create and return a new player, with parameters given by the user
def create_player(num):
    p_type = input("Who plays p" + str(num) + "? human (h) or bot(r/ab/mc): ")
    id = "d"
    budget = 50
    c_param = 1
    eval = "r"
    if p_type == "ab":
        id = input("Iterative deepening (y/n): ")
        if id == "y":
            print("Automatically using Dijkstra...")
            budget = int(input("Iterative deepening budget in seconds: "))
            eval = "d"
        else:
            budget = int(input("No iterative deepening - Budget of alpha-beta in terms of depth: "))
            eval = input("Evaluation Heuristic of board ([r]andom/[d]ijkstra): ")

    elif p_type == "mc":
        budget = int(input("Number of simulations: "))
        c_param = float(input("C parameter value (reward for exploration): "))

    return Player(num, p_type, eval, budget, c_param, id)

# Main game loop
def playgame(p1,p2,board_size):
    game = Hexboard(int(board_size))
    botmode = True
    if p1.get_type() == 'h' or p2.get_type() == 'h':
        botmode = False
    turn = 0
    current_player = p1

    while (game.is_game_over() == False):
        if game.get_turn() == Hexboard.BLUE:
            current_player = p1
        else:
            current_player = p2
        if not botmode:
            game.print()
            if current_player.get_num() == 1:
                print("It's player 1's turn (blue).")
            else:
                print("It's player 2's turn (red).")
        if current_player.get_type() == "h":
            human_move(game)
        elif current_player.get_type() == "ab":
            eval = iterative_deepening(copy.deepcopy(game), current_player)
            if botmode:
                pass
                # print("Visited " + str(eval[3]) + " nodes, with " + str(eval[4]) + " cutoffs.")
            game.place(eval[1:3], game.get_turn())
        elif current_player.get_type() == "mc":
            mctree = Node(copy.deepcopy(game), None)
            game = monte_carlo_tree_search(mctree, current_player) # MC returns the game and we 'take it over'
        else: #random
            random_move(game)

        current_player = game.get_turn()
        turn+=1

    if not botmode:
        print("Game finished with the following board")
        game.print()

    return game.get_result()




def normal_play():
    player_1 = create_player(1)
    player_2 = create_player(2)
    board_size = input("On what board size would you like to play hex?")

    for i in range(10):
        result = playgame(player_1, player_2, board_size)
        print("result", result)
        if player_1 != player_2:
            if result == Hexboard.BLUE: #p1 wins
                a, b = rate_1vs1(player_1.get_rating(), player_2.get_rating())
                player_1.set_rating(a), player_2.set_rating(b)
            elif result == Hexboard.RED: #p2 wins
                a, b = rate_1vs1(player_2.get_rating(), player_1.get_rating())
                player_1.set_rating(b), player_2.set_rating(a)
            else: #draw
                a, b = rate_1vs1(player_1.get_rating(), player_2.get_rating(), True)
                player_1.set_rating(a), player_2.set_rating(b)
        else:
            print("these are the same, elo only updates if the players are different")

    print("final elo's", player_1.get_rating().mu, player_2.get_rating().mu)


def create_player_code(player):
    code=""
    if player.get_type() == 'r':
        code += "random"
    elif player.get_type() == 'ab':
        code += "ab-"
        if player.get_eval() == 'r':
            code += 'r-'
        elif player.get_eval() == 'd':
            code += 'd-'
        if player.get_id() == 'y':
            code += 'idtt-'
        code += str(player.get_budget())
    elif player.get_type() == 'mc':
        code += "mc-"
        code += str(player.get_budget()) +"-"
        cpar = player.get_c_param()
        code += "% 12.3f" % cpar
    return code

def run_experiments(board_size):
    board_size=4
    seed(time.time())
    # num, type='r', eval='r', budget=50, c_param=1, id='n'

    # experiment 1
    players = {"player_1": (Player(1, 'ab', 'r', 3), [25], [8.333333333333334]),
                 "player_2": (Player(1, 'ab', 'd', 3), [25], [8.333333333333334]),
                 "player_3": (Player(1, 'ab', 'd', 4), [25], [8.333333333333334])}
    player_keys = ['player_1', 'player_2', 'player_3']

    # experiment 2
    # players = {"player_1": (Player(1, 'ab', 'r', 3), [25], [8.333333333333334]),
    #             "player_2": (Player(1, 'ab', 'd', 4), [25], [8.333333333333334]),
    #             "player_3": (Player(1, 'ab', 'd', 5, id='y'), [25],[8.333333333333334] )}
    # player_keys = ['player_1', 'player_2', 'player_3']

    # experiment 3
    # players = {"player_1": (Player(1, 'ab', 'd', 5, id='y'), [25], [8.333333333333334]),
    #             "player_2": (Player(1, 'mc', budget=100, c_param=1), [25], [8.333333333333334]),
    #             "player_3": (Player(1, 'mc', budget=100, c_param=2), [25], [8.333333333333334]),
    #             "player_4": (Player(1, 'mc', budget=100, c_param=0.5), [25], [8.333333333333334])}
    # player_keys = ['player_1', 'player_2', 'player_3', 'player_4']

    # experiment 4
    # players = {"player_1": (Player(1, 'ab', 'd', 5, id='y'), [25], [8.333333333333334]),
    #             "player_2": (Player(1, 'mc', budget=50), [25], [8.333333333333334]),
    #             "player_3": (Player(1, 'mc', budget=100), [25], [8.333333333333334]),
    #             "player_4": (Player(1, 'mc', budget=200), [25], [8.333333333333334])}
    # player_keys = ['player_1', 'player_2', 'player_3', 'player_4']

    counter = 0
    for i in tqdm(range(15)):
        combinations = []
        results = []
        player_combinations = []
        #create all possible combinations
        for subset in list(itertools.combinations(player_keys, 2)):
            for p in list(itertools.permutations(subset)):
                player_1 = players[p[0]][0]
                player_2 = players[p[1]][0]
                result = playgame(player_1, player_2, board_size)
                if result == Hexboard.BLUE: #p1 wins
                    a, b = rate_1vs1(player_1.get_rating(), player_2.get_rating())
                    player_1.set_rating(a), player_2.set_rating(b)
                elif result == Hexboard.RED: #p2 wins
                    a, b = rate_1vs1(player_2.get_rating(), player_1.get_rating())
                    player_1.set_rating(b), player_2.set_rating(a)
                else: #draw
                    a, b = rate_1vs1(player_1.get_rating(), player_2.get_rating(), True)
                    player_1.set_rating(a), player_2.set_rating(b)
                players[p[0]][1].append(player_1.get_rating().mu)
                players[p[1]][1].append(player_2.get_rating().mu)
                players[p[0]][2].append(player_1.get_rating().sigma)
                players[p[1]][2].append(player_2.get_rating().sigma)
                counter += 1
    print("played " + str(counter) + " games")
    plt.plot(figsize=(8, 5))
    plt.xlabel('Total games played')
    plt.ylabel('Trueskill Rating')
    for p in player_keys:
        print(p)
        print(players[p][0].get_rating())
        print(create_player_code(players[p][0]))
        plt.plot(np.arange(len(players[p][1])), np.asarray(players[p][1]), label=create_player_code(players[p][0]))
        plt.fill_between(np.arange(len(players[p][1])), np.asarray(players[p][1])-np.asarray(players[p][2]),
            np.asarray(players[p][1])+np.asarray(players[p][2]), alpha = 0.2)
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.clf()

# Change this: run experiments is for running experiments, normal_play for an interface that allows the user to choose players
# run_experiments(board_size=4)
normal_play()
