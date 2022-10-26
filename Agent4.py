"""
Your submission should be a python file with the last 'def' accepting an observation and returning an action. You can also upload multiple files in a zip/gz/7z archive with a main.py at the top level.
"""

import random
import numpy as np

# Gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, mark, config):
    next_grid = grid.copy()
    for row in range(config.rows-1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid

# Helper function for get_heuristic: checks if window satisfies heuristic conditions
def check_window(window, num_discs, piece, config):
    return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)
    
# Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
def find_spots_4(grid, num_discs, piece, config):
    good_spots = []
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if check_window(window, num_discs, piece, config):
                for i,spot in enumerate(window):
                    r,c = row,col+i
                    assert grid[r,c] == spot
                    if (r,c) not in good_spots and spot == 0:
                        good_spots.append((r,c))
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if check_window(window, num_discs, piece, config):
                for i,spot in enumerate(window):
                    r,c = row+i,col
                    assert grid[r,c] == spot
                    if (r,c) not in good_spots and spot == 0:
                        good_spots.append((r,c))
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                for i,spot in enumerate(window):
                    r,c = row+i,col+i
                    assert grid[r,c] == spot
                    if (r,c) not in good_spots and spot == 0:
                        good_spots.append((r,c))
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                for i,spot in enumerate(window):
                    r,c = row-i,col+i
                    assert grid[r,c] == spot
                    if (r,c) not in good_spots and spot == 0:
                        good_spots.append((r,c))
    return good_spots

# Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
def count_windows_4(grid, num_discs, piece, config):
    num_windows = 0
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    return num_windows

# Calculate spots from 0 + Adjacency bonus
def get_heuristic_4(grid, mark, config):
    score = 0
    for i,num in enumerate(range(0, config.inarow-1)):
        weight = 10**i 
        num_wins = count_windows_4(grid, num, mark, config)
        num_wins_opp = count_windows_4(grid, num, mark%2+1, config)
        score += weight * num_wins - 5 * weight * num_wins_opp
    weight = 10**(config.inarow-1) 
    good_spots = find_spots_4(grid, num, mark, config)
    good_spots_opp = find_spots_4(grid, num, mark%2+1, config)
    score += weight * len(good_spots) - 5 * weight * len(good_spots_opp)
    weight = 10**(config.inarow) 
    # Adjacency bonus
    prev_c, prev_r = -1, -1 
    for c,r in sorted([(c,r) for r,c in good_spots]):
        if c == prev_c and r == prev_r + 1:
            score += weight
        prev_c, prev_r = c, r
    prev_c, prev_r = -1, -1 
    for c,r in sorted([(c,r) for r,c in good_spots_opp]):
        if c == prev_c and r == prev_r + 1:
            score -= weight
        prev_c, prev_r = c, r
    weight = 10**(config.inarow+1) 
    num_wins = count_windows_4(grid, config.inarow, mark, config)
    num_wins_opp = count_windows_4(grid, config.inarow, mark%2+1, config)
    score += weight * num_wins - 5 * weight * num_wins_opp
    return score

# Uses minimax to calculate value of dropping piece in selected column
def score_move_4(grid, col, mark, config, nsteps):
    next_grid = drop_piece(grid, col, mark, config)
    score = minimax_4(next_grid, nsteps-1, False, mark, config)
    return score

# Helper function for minimax: checks if agent or opponent has four in a row in the window
def is_terminal_window(window, config):
    return window.count(1) == config.inarow or window.count(2) == config.inarow

# Helper function for minimax: checks if game has ended
def is_terminal_node(grid, config):
    # Check for draw 
    if list(grid[0, :]).count(0) == 0:
        return True
    # Check for win: horizontal, vertical, or diagonal
    # horizontal 
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if is_terminal_window(window, config):
                return True
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if is_terminal_window(window, config):
                return True
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return True
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return True
    return False

# Minimax implementation
def minimax_4(node, depth, maximizingPlayer, mark, config):
    is_terminal = is_terminal_node(node, config)
    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
    if depth == 0 or is_terminal:
        return get_heuristic_4(node, mark, config)
    if maximizingPlayer:
        value = -np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark, config)
            value = max(value, minimax_4(child, depth-1, False, mark, config))
        return value
    else:
        value = np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark%2+1, config)
            value = min(value, minimax_4(child, depth-1, True, mark, config))
        return value
    
# How deep to make the game tree: higher values take longer to run!
N_STEPS = 3

def agent(obs, config):
    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move_4(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)
