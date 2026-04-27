"""
backend/solver.py
=================
Sudoku solver — Backtracking + MRV (Minimum Remaining Values) heuristic.
Also exposes validate_board() for the live /api/validate endpoint.
"""


def _is_valid(board, idx, val):
    row, col = divmod(idx, 9)
    for c in range(9):
        if c != col and board[row * 9 + c] == val:
            return False
    for r in range(9):
        if r != row and board[r * 9 + col] == val:
            return False
    br, bc = (row // 3) * 3, (col // 3) * 3
    for dr in range(3):
        for dc in range(3):
            ni = (br + dr) * 9 + (bc + dc)
            if ni != idx and board[ni] == val:
                return False
    return True


def _candidates(board, idx):
    if board[idx] != 0:
        return set()
    return {n for n in range(1, 10) if _is_valid(board, idx, n)}


def _pick_cell(board):
    best_idx, best_n = -1, 10
    for i in range(81):
        if board[i] == 0:
            cands = _candidates(board, i)
            if len(cands) == 0:
                return i, set()
            if len(cands) < best_n:
                best_idx, best_n = i, len(cands)
    return best_idx, (_candidates(board, best_idx) if best_idx >= 0 else set())


def _backtrack(board):
    idx, cands = _pick_cell(board)
    if idx == -1:
        return True
    for n in sorted(cands):
        board[idx] = n
        if _backtrack(board):
            return True
        board[idx] = 0
    return False


def _find_conflicts(board):
    seen, conflicts = set(), []
    for i in range(81):
        v = board[i]
        if not v:
            continue
        row, col = divmod(i, 9)
        br, bc   = (row // 3) * 3, (col // 3) * 3
        peers = (
            [row * 9 + c for c in range(9) if c != col] +
            [r * 9 + col for r in range(9) if r != row] +
            [(br + dr) * 9 + (bc + dc)
             for dr in range(3) for dc in range(3)
             if (br + dr) * 9 + (bc + dc) != i]
        )
        for ni in peers:
            if board[ni] == v:
                key = (min(i, ni), max(i, ni))
                if key not in seen:
                    seen.add(key)
                    conflicts.append(list(key))
    return conflicts


def solve_sudoku(flat_board: list) -> dict:
    """
    Solve a sudoku puzzle.
    Returns dict with keys: solved, board, conflicts, error.
    """
    conflicts = _find_conflicts(flat_board)
    if conflicts:
        return {
            "solved":    False,
            "board":     flat_board,
            "conflicts": conflicts,
            "error":     "Conflicting digits detected — see highlighted cells.",
        }
    copy   = flat_board[:]
    solved = _backtrack(copy)
    return {
        "solved":    solved,
        "board":     copy if solved else flat_board,
        "conflicts": [],
        "error":     None if solved else
                     "No valid solution exists. Check for incorrect given digits.",
    }


def validate_board(flat_board: list) -> dict:
    """Live validation — returns conflict list without solving."""
    conflicts = _find_conflicts(flat_board)
    return {"valid": len(conflicts) == 0, "conflicts": conflicts}
