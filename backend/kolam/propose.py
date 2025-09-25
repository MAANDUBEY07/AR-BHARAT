from __future__ import annotations
from typing import List, Tuple
import random

# --- Connectivity and symmetry tables (1-based ids 1..16) ---
# pt_dn / pt_rt flags indicate whether a prototype connects down/right
pt_dn = [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1]
pt_rt = [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]

# Mates indexed by flag 0/1
mate_pt_dn = {
    0: [i for i in range(2, 17) if i not in [2, 3, 5, 6, 9, 10, 12]],
    1: [2, 3, 5, 6, 9, 10, 12],
}
mate_pt_rt = {
    0: [i for i in range(2, 17) if i not in [2, 3, 4, 6, 7, 11, 13]],
    1: [2, 3, 4, 6, 7, 11, 13],
}

# Horizontal/Vertical inverse maps (1-based mapping arrays)
h_inv = [1, 2, 5, 4, 3, 9, 8, 7, 6, 10, 11, 12, 15, 14, 13, 16]
v_inv = [1, 4, 3, 2, 5, 7, 6, 9, 8, 10, 11, 14, 13, 12, 15, 16]

# Self-symmetric ids (where inv(id) == id)
h_self = [i for i in range(1, 17) if h_inv[i-1] == i]
v_self = [i for i in range(1, 17) if v_inv[i-1] == i]

# Additional maps used in 2D proposal
flip_90 = [1, 3, 2, 5, 4, 6, 9, 8, 7, 11, 10, 13, 12, 15, 14, 16]

diagsym = [1, 6, 8, 16]


def _rand_pick(valids: List[int]) -> int:
    if not valids:
        return 1
    return random.choice(valids)


def propose_kolam1D(n: int) -> List[List[int]]:
    """Python port of MATLAB propose_kolam1D.
    Returns an n x n matrix with ids 1..16 (1 means blank/no stroke).
    """
    if n <= 0:
        return [[1]]
    odd = (n % 2 != 0)
    hp = (n - 1) // 2 if odd else (n // 2)

    # Allocate Mat of size (hp+2) x (hp+2) pre-filled with 1s (MATLAB grows later; we pre-size)
    N = hp + 2
    Mat = [[1 for _ in range(N)] for __ in range(N)]

    # Fill interior (2..hp+1, 2..hp+1)
    for i in range(1, hp + 1):  # Python 0-based; corresponds to MATLAB i=2..hp+1
        for j in range(1, hp + 1):  # MATLAB j=2..hp+1
            up_val = Mat[i - 1][j]
            lt_val = Mat[i][j - 1]
            valid_by_up = mate_pt_dn[pt_dn[up_val - 1]]
            valid_by_lt = mate_pt_rt[pt_rt[lt_val - 1]]
            valids = [x for x in valid_by_up if x in valid_by_lt]
            Mat[i][j] = _rand_pick(valids)

    # Borders defaults already 1
    # Bottom row (hp+2, j=2..hp+1), constrain v_self
    last = N - 1
    for j in range(1, hp + 1):
        up_val = Mat[last - 1][j]
        lt_val = Mat[last][j - 1]
        valid_by_up = mate_pt_dn[pt_dn[up_val - 1]]
        valid_by_lt = mate_pt_rt[pt_rt[lt_val - 1]]
        valids = [x for x in valid_by_up if x in valid_by_lt and x in v_self]
        Mat[last][j] = _rand_pick(valids)

    # Right column (i=2..hp+1, hp+2), constrain h_self
    for i in range(1, hp + 1):
        up_val = Mat[i - 1][last]
        lt_val = Mat[i][last - 1]
        valid_by_up = mate_pt_dn[pt_dn[up_val - 1]]
        valid_by_lt = mate_pt_rt[pt_rt[lt_val - 1]]
        valids = [x for x in valid_by_up if x in valid_by_lt and x in h_self]
        Mat[i][last] = _rand_pick(valids)

    # Corner (hp+2, hp+2)
    up_val = Mat[last - 1][last]
    lt_val = Mat[last][last - 1]
    valid_by_up = mate_pt_dn[pt_dn[up_val - 1]]
    valid_by_lt = mate_pt_rt[pt_rt[lt_val - 1]]
    valids = [x for x in valid_by_up if x in valid_by_lt and x in h_self and x in v_self]
    Mat[last][last] = _rand_pick(valids)

    # Extract building blocks
    # Mat1 = Mat(2:hp+1, 2:hp+1)
    Mat1 = [row[1:hp + 1] for row in Mat[1:hp + 1]]
    # Mat2 = h_inv(Mat1(:, end:-1:1)) => flip LR then map via h_inv
    Mat1_lr = [list(reversed(row)) for row in Mat1]
    Mat2 = [[h_inv[val - 1] for val in row] for row in Mat1_lr]
    # Mat3 = v_inv(Mat1(end:-1:1, :)) => flip UD then map via v_inv
    Mat1_ud = list(reversed(Mat1))
    Mat3 = [[v_inv[val - 1] for val in row] for row in Mat1_ud]
    # Mat4 = v_inv(Mat2(end:-1:1, :))
    Mat2_ud = list(reversed(Mat2))
    Mat4 = [[v_inv[val - 1] for val in row] for row in Mat2_ud]

    if odd:
        # Assemble with border strips
        top_right_col = [Mat[i][last] for i in range(1, N - 1)]  # Mat(2:end-1, end)
        bottom_row_rev = [Mat[last][j] for j in range(N - 2, 0, -1)]  # Mat(end, (end-1):-1:2)
        bottom_row_inv = [h_inv[val - 1] for val in bottom_row_rev]
        right_col_rev = [Mat[i][last] for i in range(N - 2, 0, -1)]  # Mat((end-1):-1:2, end)
        right_col_vinv = [v_inv[val - 1] for val in right_col_rev]

        # Build full M
        # [ Mat1   top_right_col   Mat2 ]
        # [ bottom_row ...                 ]
        # [ Mat3   right_col_vinv'  Mat4 ]
        left = [row[:] for row in Mat1]
        mid_col = [[val] for val in top_right_col]
        right = [row[:] for row in Mat2]
        upper = [l + m + r for l, m, r in zip(left, mid_col, right)]

        middle = bottom_row_inv  # 1 x (cols)

        left2 = [row[:] for row in Mat3]
        mid_col2 = [[val] for val in right_col_vinv]
        right2 = [row[:] for row in Mat4]
        lower = [l + m + r for l, m, r in zip(left2, mid_col2, right2)]

        M = upper + [middle] + lower
    else:
        # Even case: simple 4 quadrants
        left = [row[:] for row in Mat1]
        right = [row[:] for row in Mat2]
        upper = [l + r for l, r in zip(left, right)]
        left2 = [row[:] for row in Mat3]
        right2 = [row[:] for row in Mat4]
        lower = [l + r for l, r in zip(left2, right2)]
        M = upper + lower

    return M


def propose_kolam2D(n: int) -> List[List[int]]:
    """Python port of MATLAB propose_kolam2D.
    Returns an n x n matrix with ids 1..16 (1 means blank/no stroke).
    """
    if n <= 0:
        return [[1]]
    odd = (n % 2 != 0)
    hp = (n - 1) // 2 if odd else (n // 2)

    N = hp + 2
    Mat = [[1 for _ in range(N)] for __ in range(N)]
    # Force Mat(1,hp+2)=1 already

    # Fill diagonal first
    for i in range(1, hp + 1):  # MATLAB i=2..hp+1
        if i == 1:
            # Corresponds to MATLAB i==2 special case (we are 0-based)
            Mat[1][1] = random.choice(diagsym)
        else:
            up_val = Mat[i - 1][i]
            lt_val = Mat[i][i - 1]
            valid_by_up = mate_pt_dn[pt_dn[up_val - 1]]
            valid_by_lt = mate_pt_rt[pt_rt[lt_val - 1]]
            valids = [x for x in valid_by_up if x in valid_by_lt and x in diagsym]
            Mat[i][i] = _rand_pick(valids)
        # Fill upper triangle j=(i+1)..(hp+1) and mirror with flip_90
        for j in range(i + 1, hp + 1):
            up_val = Mat[i - 1][j]
            lt_val = Mat[i][j - 1]
            valid_by_up = mate_pt_dn[pt_dn[up_val - 1]]
            valid_by_lt = mate_pt_rt[pt_rt[lt_val - 1]]
            valids = [x for x in valid_by_up if x in valid_by_lt]
            Mat[i][j] = _rand_pick(valids)
            Mat[j][i] = flip_90[Mat[i][j] - 1]

    last = N - 1
    # Right column and bottom row with symmetry constraints
    for i in range(1, hp + 1):
        up_val = Mat[i - 1][last]
        lt_val = Mat[i][last - 1]
        valid_by_up = mate_pt_dn[pt_dn[up_val - 1]]
        valid_by_lt = mate_pt_rt[pt_rt[lt_val - 1]]
        valids = [x for x in valid_by_up if x in valid_by_lt and x in h_self]
        Mat[i][last] = _rand_pick(valids)
        Mat[last][i] = flip_90[Mat[i][last] - 1]

    # Corner
    up_val = Mat[last - 1][last]
    lt_val = Mat[last][last - 1]
    valid_by_up = mate_pt_dn[pt_dn[up_val - 1]]
    valid_by_lt = mate_pt_rt[pt_rt[lt_val - 1]]
    valids = [x for x in valid_by_up if x in valid_by_lt and x in h_self and x in v_self]
    Mat[last][last] = _rand_pick(valids)

    # Quadrants
    Mat1 = [row[1:hp + 1] for row in Mat[1:hp + 1]]
    Mat1_lr = [list(reversed(row)) for row in Mat1]
    Mat2 = [[h_inv[val - 1] for val in row] for row in Mat1_lr]
    Mat1_ud = list(reversed(Mat1))
    Mat3 = [[v_inv[val - 1] for val in row] for row in Mat1_ud]
    Mat2_ud = list(reversed(Mat2))
    Mat4 = [[v_inv[val - 1] for val in row] for row in Mat2_ud]

    if odd:
        top_right_col = [Mat[i][last] for i in range(1, N - 1)]
        bottom_row_rev = [Mat[last][j] for j in range(N - 2, 0, -1)]
        bottom_row_inv = [h_inv[val - 1] for val in bottom_row_rev]
        right_col_rev = [Mat[i][last] for i in range(N - 2, 0, -1)]
        right_col_vinv = [v_inv[val - 1] for val in right_col_rev]

        left = [row[:] for row in Mat1]
        mid_col = [[val] for val in top_right_col]
        right = [row[:] for row in Mat2]
        upper = [l + m + r for l, m, r in zip(left, mid_col, right)]

        middle = bottom_row_inv

        left2 = [row[:] for row in Mat3]
        mid_col2 = [[val] for val in right_col_vinv]
        right2 = [row[:] for row in Mat4]
        lower = [l + m + r for l, m, r in zip(left2, mid_col2, right2)]

        M = upper + [middle] + lower
    else:
        left = [row[:] for row in Mat1]
        right = [row[:] for row in Mat2]
        upper = [l + r for l, r in zip(left, right)]
        left2 = [row[:] for row in Mat3]
        right2 = [row[:] for row in Mat4]
        lower = [l + r for l, r in zip(left2, right2)]
        M = upper + lower

    return M