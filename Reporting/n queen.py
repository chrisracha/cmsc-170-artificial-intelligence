import random

def create_random_board(n):
    return [random.randint(0, n - 1) for _ in range(n)]

def count_conflicts(board):
    n = len(board)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                conflicts += 1
    return conflicts

def get_best_neighbor(board):
    n = len(board)
    best_board = board[:]
    best_conflicts = count_conflicts(board)
    
    for col in range(n):
        original_row = board[col]
        for new_row in range(n):
            if new_row != original_row:
                board[col] = new_row
                conflicts = count_conflicts(board)
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_board = board[:]
        board[col] = original_row
    
    return best_board, best_conflicts

def hill_climbing_solve(n):
    board = create_random_board(n)
    conflicts = count_conflicts(board)
    
    while conflicts > 0:
        new_board, new_conflicts = get_best_neighbor(board)
        if new_conflicts >= conflicts:
            return board, False
        board = new_board
        conflicts = new_conflicts
    
    return board, True

def solve_with_restarts(n, max_restarts=10):
    for restart in range(max_restarts):
        solution, solved = hill_climbing_solve(n)
        if solved:
            return solution
    return None

def print_board(board):
    n = len(board)
    for row in range(n):
        for col in range(n):
            print("Q" if board[col] == row else ".", end=" ")
        print()

n = 4
solution, solved = hill_climbing_solve(n)
print(f"Solution: {solution}")
print_board(solution)
print(f"Conflicts: {count_conflicts(solution)}")
if solved:
    print("Found solution!")
else:
    print("Stuck at local maximum")