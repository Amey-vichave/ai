# ai
1. DFS for 4-Queens Problem
python
CopyEdit
def is_safe(state, row, col):
    for r in range(row):
        c = state[r]
        if c == col or abs(c - col) == abs(r - row):
            return False
    return True

def dfs_4_queens(row=0, state=[]):
    if row == 4:
        print("Solution:", state)
        return
    for col in range(4):
        if is_safe(state, row, col):
            dfs_4_queens(row + 1, state + [col])

dfs_4_queens()
Explanation:
This uses DFS to explore placing queens row by row.
It avoids placing queens in unsafe positions using is_safe.

2. DFS for 8-Puzzle Problem
python
CopyEdit
from collections import deque

def print_board(state):
    for i in range(0, 9, 3):
        print(state[i:i+3])
    print()

def get_moves(state):
    moves = []
    i = state.index(0)
    row, col = i // 3, i % 3
    directions = {'up': -3, 'down': 3, 'left': -1, 'right': 1}
    
    for move, delta in directions.items():
        new_i = i + delta
        if move == 'left' and col == 0: continue
        if move == 'right' and col == 2: continue
        if move == 'up' and row == 0: continue
        if move == 'down' and row == 2: continue
        new_state = state[:]
        new_state[i], new_state[new_i] = new_state[new_i], new_state[i]
        moves.append(new_state)
    return moves

def dfs_8_puzzle(start, goal):
    stack = [(start, [])]
    visited = set()

    while stack:
        state, path = stack.pop()
        if tuple(state) in visited:
            continue
        visited.add(tuple(state))

        if state == goal:
            print("Goal reached!")
            for p in path + [state]:
                print_board(p)
            return

        for move in get_moves(state):
            stack.append((move, path + [state]))

start_state = [1, 2, 3, 4, 0, 5, 6, 7, 8]
goal_state = [1, 2, 3, 4, 5, 0, 6, 7, 8]
dfs_8_puzzle(start_state, goal_state)


3. A Algorithm for 8 Puzzle*
python
CopyEdit
import heapq

def heuristic(state, goal):
    return sum(s != g and s != 0 for s, g in zip(state, goal))

def get_neighbors(state):
    neighbors = []
    i = state.index(0)
    row, col = i // 3, i % 3
    directions = {'up': -3, 'down': 3, 'left': -1, 'right': 1}
    
    for move, delta in directions.items():
        new_i = i + delta
        if move == 'left' and col == 0: continue
        if move == 'right' and col == 2: continue
        if move == 'up' and row == 0: continue
        if move == 'down' and row == 2: continue
        new_state = state[:]
        new_state[i], new_state[new_i] = new_state[new_i], new_state[i]
        neighbors.append(new_state)
    return neighbors

def a_star(start, goal):
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start, []))
    visited = set()

    while open_list:
        _, cost, state, path = heapq.heappop(open_list)
        if tuple(state) in visited:
            continue
        visited.add(tuple(state))
        if state == goal:
            print("A* Path:")
            for p in path + [state]:
                print(p)
            return
        for neighbor in get_neighbors(state):
            if tuple(neighbor) not in visited:
                heapq.heappush(open_list, (cost + 1 + heuristic(neighbor, goal), cost + 1, neighbor, path + [state]))

start = [1, 2, 3, 4, 0, 5, 6, 7, 8]
goal = [1, 2, 3, 4, 5, 0, 6, 7, 8]
a_star(start, goal)

4. Minimax Algorithm (Tic-Tac-Toe)
python
CopyEdit
def is_winner(board, player):
    win = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    return any(all(board[i] == player for i in combo) for combo in win)

def minimax(board, is_max):
    if is_winner(board, 'X'): return 1
    if is_winner(board, 'O'): return -1
    if ' ' not in board: return 0

    if is_max:
        best = -float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                best = max(best, minimax(board, False))
                board[i] = ' '
        return best
    else:
        best = float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                best = min(best, minimax(board, True))
                board[i] = ' '
        return best

# Example Usage
board = ['X', 'O', 'X', ' ', 'O', ' ', ' ', ' ', ' ']
print("Best value for current board:", minimax(board, True))

5. Alpha-Beta Pruning
python
CopyEdit
def alphabeta(board, depth, alpha, beta, is_max):
    if is_winner(board, 'X'): return 1
    if is_winner(board, 'O'): return -1
    if ' ' not in board: return 0

    if is_max:
        max_eval = -float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                eval = alphabeta(board, depth+1, alpha, beta, False)
                board[i] = ' '
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha: break
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                eval = alphabeta(board, depth+1, alpha, beta, True)
                board[i] = ' '
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: break
        return min_eval

board = ['X', 'O', 'X', ' ', 'O', ' ', ' ', ' ', ' ']
print("Best value with Alpha-Beta:", alphabeta(board, 0, -float('inf'), float('inf'), True))

6. Hill Climbing (Simple Function Maximize)
python
CopyEdit
import random

def objective(x):
    return -(x - 3)**2 + 9  # Maximum at x = 3

def hill_climb():
    x = random.uniform(-10, 10)
    step = 0.1
    for _ in range(100):
        new_x = x + random.uniform(-step, step)
        if objective(new_x) > objective(x):
            x = new_x
    print("Best x:", x)
    print("Best value:", objective(x))

hill_climb()

7 & 8. FOPL Inference (Conceptual Only)
You'd represent sentences in logic and draw a resolution tree manually for wumpus-type problems. Python doesn't natively handle FOPL easily—use Prolog or tools like sympy for symbolic logic.

9. Reinforcement Learning Reward System for Tic-Tac-Toe
python
CopyEdit
import random

Q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.1

def get_Q(state, action):
    return Q.get((state, action), 0)

def choose_action(state, actions):
    if random.random() < epsilon:
        return random.choice(actions)
    qs = [get_Q(state, a) for a in actions]
    return actions[qs.index(max(qs))]

def update_Q(state, action, reward, next_state, next_actions):
    max_next_Q = max([get_Q(next_state, a) for a in next_actions], default=0)
    Q[(state, action)] = get_Q(state, action) + alpha * (reward + gamma * max_next_Q - get_Q(state, action))
Note: This is just the learning logic, you’d need to wrap it with a Tic-Tac-Toe environment.

10. Coin Toss with Bayesian Updating
python
CopyEdit
def bayesian_update(prior_heads, prior_tails, data):
    for toss in data:
        if toss == 'H':
            prior_heads += 1
        else:
            prior_tails += 1
    total = prior_heads + prior_tails
    return prior_heads / total, prior_tails / total

# Start with prior of 1 head, 1 tail
data = ['H', 'H', 'T', 'H', 'T']
posterior = bayesian_update(1, 1, data)
print("Probability of Head:", posterior[0])
print("Probability of Tail:", posterior[1])

11. Tokenization (Split Sentences into Words)
python
CopyEdit
sentence = "Artificial Intelligence is fun!"
tokens = sentence.split()
print(tokens)

12. Stopword Removal
python
CopyEdit
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

sentence = "This is a sample sentence with common words"
words = sentence.split()
filtered = [word for word in words if word.lower() not in stopwords.words('english')]
print(filtered)
