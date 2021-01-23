"""20x24 representation of a Rubick's cube"""
import numpy as np
from functools import wraps, partial

# Sides are defined by letters: U, D, F, B, L, R
#
#
#                  +----+----+----+
#                  | B2 | B1 | B0 |
#                  +----+----+----+
#                  | B3 |    | B7 |
#                  +----+----+----+
#                  | B4 | B5 | B6 |
#   +----+----+----+----+----+----+----+----+----+----+----+----+
#   | L0 | L7 | L6 | U6 | U5 | U4 | R4 | R3 | R2 | D4 | D5 | D6 |
#   +----+----+----+----+----+----+----+----+----+----+----+----+
#   | L1 |    | L5 | U7 |    | U3 | R5 |    | R1 | D3 |    | D7 |
#   +----+----+----+----+----+----+----+----+----+----+----+----+
#   | L2 | L3 | L4 | U0 | U1 | U2 | R6 | R7 | R0 | D2 | D1 | D0 |
#   +----+----+----+----+----+----+----+----+----+----+----+----+
#                  | F6 | F5 | F4 |
#                  +----+----+----+
#                  | F7 |    | F3 |
#                  +----+----+----+
#                  | F0 | F1 | F2 |
#                  +----+----+----+
#
#
# Example
# =======
# Possible locations for U0:
#
# U0, U2, U4, U6,  (U, U, U, U)
# L2, L4, L6, L0,  (F', L', L', L')
# R6, R0, R2, R4,  (F, R', R', R')
# B4, B6, B0, B2,  (L', B', B', B')
# D2, D0, D6, D4,  (F, F, D', D', D')
#
#
# Corner pivots (8)
# =================
# [U0, U2, U4, U6, D0, D2, D4, D6]
#
#
# Edge pivots (12)
# ================
# [U1, U3, U5, U7, F7, F3, B7, B3, D1, D3, D5, D7]
#
#
# Locations for corner pivots (24)
# ================================
# [U0, U2, U4, U6,
#  F0, F2, F4, F6,
#  B4, B6, B0, B2,
#  L2, L4, L6, L0,
#  R6, R4, R2, R0,
#  D6, D4, D2, D0]
#
#
# Location for edge pivots (24)
# =============================
# [U1, U3, U5, U7,
#  F1, F3, F5, F7,
#  B5, B7, B1, B3,
#  L3, L5, L7, L1,
#  R7, R1, R3, R5,
#  D5, D3, D1, D7]
#
#
# Corner table (8 x 24)
# =====================
#
#    | U0, U2, U4, U6, F0, F2, F4, F6, B4, B6, B0, B2, L2, L4, L6, L0, R6, R4, R2, R0, D6, D4, D2, D0
# ---+-----------------------------------------------------------------------------------------------
# U0 |
# U2 |
# U4 |
# U6 |
# D0 |
# D2 |
# D4 |
# D6 |
#
#
# Edge table (12 x 24)
# ====================
#
#    | U1, U3, U5, U7, F1, F3, F5, F7, B5, B7, B1, B3, L3, L5, L7, L1, R7, R1, R3, R5, D5, D3, D1, D7
# ---+-----------------------------------------------------------------------------------------------
# U1 |
# U3 |
# U5 |
# U7 |
# F7 |
# F3 |
# B7 |
# B3 |
# D7 |
# D1 |
# D3 |
# D5 |


### =============================== GLOBALS =============================== ###

_CORNER_TABLE_COL_INDEX = {k: v for v, k in enumerate(
    'u0 u2 u4 u6 f0 f2 f4 f6 b4 b6 b0 b2 l2 l4 l6 l0 r6 r4 r2 r0 d6 d4 d2 d0'.split()
)}
assert(len(_CORNER_TABLE_COL_INDEX) == 24)

_CORNER_TABLE_ROW_INDEX = {k: v for v, k in enumerate(
    'u0 u2 u4 u6 d0 d2 d4 d6'.split()
)}
assert(len(_CORNER_TABLE_ROW_INDEX) == 8)

_EDGE_TABLE_COL_INDEX = {k: v for v, k in enumerate(
    'u1 u3 u5 u7 f1 f3 f5 f7 b5 b7 b1 b3 l3 l5 l7 l1 r7 r1 r3 r5 d5 d3 d1 d7'.split()
)}
assert(len(_EDGE_TABLE_COL_INDEX) == 24)

_EDGE_TABLE_ROW_INDEX = {k: v for v, k in enumerate(
    'u1 u3 u5 u7 f7 f3 b7 b3 d7 d1 d3 d5'.split()
)}
assert(len(_EDGE_TABLE_ROW_INDEX) == 12)

_SOLVED = np.array(
    [
     # ================ Corner table =============== #
     [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # U0
     [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # U2
     [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # U4
     [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # U6
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],  # D0
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],  # D2
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],  # D4
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],  # D6
     # ================= Edge table ================ #
     [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # U1
     [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # U3
     [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # U5
     [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # U7
     [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # F7
     [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # F3
     [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # B7
     [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],  # B3
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],  # D7
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],  # D1
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],  # D3
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]   # D5
    ], dtype=np.int8
)


def solved_cube():
    """
    Returns
    -------
    The solved Rubick's cube in 20x24 representation
    """
    ccol = _CORNER_TABLE_COL_INDEX
    crow = _CORNER_TABLE_ROW_INDEX
    ecol = _EDGE_TABLE_COL_INDEX
    erow = _EDGE_TABLE_ROW_INDEX

    res = np.zeros((20, 24), dtype=np.int8)
    for x in crow.keys():
        res[crow[x], ccol[x]] = 1
    for x in erow.keys():
        res[8 + erow[x], ecol[x]] = 1
    return res

assert(np.all(solved_cube() == _SOLVED))


# U : white   0
# D : yellow  1
# F : red     2
# B : orange  3
# L : blue    4
# R : green   5
_COLOR_MAP = dict(zip('UDLRFB', range(6)))

def convert_to_color(cube):
    """
    Parameters
    ----------
    cube : 20x24 representation

    Returns
    -------
    6x3x3 classic representation with colors = (0, 1, 2, 3, 4, 5)
    """


### =============================== ACTIONS =============================== ###

def Uc(cube):
    """Rotates side U clockwise inplace"""
    # Corners
    # -------
    # [u0, u2, u4, u6] <- [u2, u4, u6, u0]
    # [f6, f4] <- [r6, r4]
    # [r6, r4] <- [b6, b4]
    # [b6, b4] <- [l6, l4]
    # [l6, l4] <- [f6, f4]
    #
    # [u0 u2 u4 u6 f6 f4 r6 r4 b6 b4 l6 l4]
    # [u2 u4 u6 u0 r6 r4 b6 b4 l6 l4 f6 f4]
    #
    # Edges
    # -----
    # [u1, u3, u5, u7] <- [u3, u5, u7, u1]
    # [f5, r5, b5, l5] <- [r5, b5, l5, f5]
    #
    # u1 u3 u5 u7 f5 r5 b5 l5
    # u3 u5 u7 u1 r5 b5 l5 f5
    cmap = _CORNER_TABLE_COL_INDEX
    emap = _EDGE_TABLE_COL_INDEX

    # Swap corners
    i = [cmap[c] for c in ('u0', 'u2', 'u4', 'u6', 'f6', 'f4', 'r6', 'r4', 'b6', 'b4', 'l6', 'l4')]
    j = [cmap[c] for c in ('u2', 'u4', 'u6', 'u0', 'r6', 'r4', 'b6', 'b4', 'l6', 'l4', 'f6', 'f4')]
    cube[:8, i] = cube[:8, j]

    # Swap edges
    i = [emap[c] for c in ('u1', 'u3', 'u5', 'u7', 'f5', 'r5', 'b5', 'l5')]
    j = [emap[c] for c in ('u3', 'u5', 'u7', 'u1', 'r5', 'b5', 'l5', 'f5')]
    cube[8:, i] = cube[8:, j]


def Ucc(cube):
    """Rotates side U counterclockwise inplace"""
    # Corners
    # -------
    # [u0, u2, u4, u6] <= [u6, u0, u2, u4]
    # [f6, f4] <= [l6, l4]
    # [r6, r4] <= [f6, f4]
    # [b6, b4] <= [r6, r4]
    # [l6, l4] <= [b6, b4]
    #
    # u0 u2 u4 u6 f6 f4 r6 r4 b6 b4 l6 l4
    # u6 u0 u2 u4 l6 l4 f6 f4 r6 r4 b6 b4
    #
    # Edges
    # -----
    # [u1, u3, u5, u7] <= [u7, u1, u3, u5]
    # [f5, r5, b5, l5] <= [l5, f5, r5, b5]
    #
    # u1 u3 u5 u7 f5 r5 b5 l5
    # u7 u1 u3 u5 l5 f5 r5 b5
    cmap = _CORNER_TABLE_COL_INDEX
    emap = _EDGE_TABLE_COL_INDEX

    # Swap corners
    i = [cmap[c] for c in ('u0', 'u2', 'u4', 'u6', 'f6', 'f4', 'r6', 'r4', 'b6', 'b4', 'l6', 'l4')]
    j = [cmap[c] for c in ('u6', 'u0', 'u2', 'u4', 'l6', 'l4', 'f6', 'f4', 'r6', 'r4', 'b6', 'b4')]
    cube[:8, i] = cube[:8, j]

    # Swap edges
    i = [emap[c] for c in ('u1', 'u3', 'u5', 'u7', 'f5', 'r5', 'b5', 'l5')]
    j = [emap[c] for c in ('u7', 'u1', 'u3', 'u5', 'l5', 'f5', 'r5', 'b5')]
    cube[8:, i] = cube[8:, j]


def Fc(cube):
    """Rotates side F clockwise inplace"""
    # Corners
    # -------
    # [f0, f2, f4, f6] <= [f2, f4, f6, f0]
    # [u0, u2] <= [l2, l4]
    # [l2, l4] <= [d2, d0]
    # [d2, d0] <= [r6, r0]
    # [r6, r0] <= [u0, u2]
    #
    # f0 f2 f4 f6 u0 u2 l2 l4 d2 d0 r6 r0
    # f2 f4 f6 f0 l2 l4 d2 d0 r6 r0 u0 u2
    #
    # Edges
    # -----
    # [f1, f3, f5, f7] <= [f3, f5, f7, f1]
    # [u1, r7, d1, l3] <= [l3, u1, r7, d1]
    #
    # f1 f3 f5 f7 u1 r7 d1 l3
    # f3 f5 f7 f1 l3 u1 r7 d1
    cmap = _CORNER_TABLE_COL_INDEX
    emap = _EDGE_TABLE_COL_INDEX

    # Swap corners
    i = [cmap[c] for c in ('f0', 'f2', 'f4', 'f6', 'u0', 'u2', 'l2', 'l4', 'd2', 'd0', 'r6', 'r0')]
    j = [cmap[c] for c in ('f2', 'f4', 'f6', 'f0', 'l2', 'l4', 'd2', 'd0', 'r6', 'r0', 'u0', 'u2')]
    cube[:8, i] = cube[:8, j]

    # Swap edges
    i = [emap[c] for c in ('f1', 'f3', 'f5', 'f7', 'u1', 'r7', 'd1', 'l3')]
    j = [emap[c] for c in ('f3', 'f5', 'f7', 'f1', 'l3', 'u1', 'r7', 'd1')]
    cube[8:, i] = cube[8:, j]


def Fcc(cube):
    """Rotates side F counterclockwise inplace"""
    # Corners
    # -------
    # [f0, f2, f4, f6] <= [f6, f0, f2, f4]
    # [u0, u2] <= [r6, r0]
    # [r6, r0] <= [d2, d0]
    # [d2, d0] <= [l2, l4]
    # [l2, l4] <= [u0, u2]
    #
    # f0 f2 f4 f6 u0 u2 r6 r0 d2 d0 l2 l4
    # f6 f0 f2 f4 r6 r0 d2 d0 l2 l4 u0 u2
    #
    # Edges
    # -----
    # [f1, f3, f5, f7] <= [f7, f1, f3, f5]
    # [u1, r7, d1, l3] <= [r7, d1, l3, u1]
    #
    # f1 f3 f5 f7 u1 r7 d1 l3
    # f7 f1 f3 f5 r7 d1 l3 u1
    cmap = _CORNER_TABLE_COL_INDEX
    emap = _EDGE_TABLE_COL_INDEX

    # Swap corners
    i = [cmap[c] for c in ('f0', 'f2', 'f4', 'f6', 'u0', 'u2', 'r6', 'r0', 'd2', 'd0', 'l2', 'l4')]
    j = [cmap[c] for c in ('f6', 'f0', 'f2', 'f4', 'r6', 'r0', 'd2', 'd0', 'l2', 'l4', 'u0', 'u2')]
    cube[:8, i] = cube[:8, j]

    # Swap edges
    i = [emap[c] for c in ('f1', 'f3', 'f5', 'f7', 'u1', 'r7', 'd1', 'l3')]
    j = [emap[c] for c in ('f7', 'f1', 'f3', 'f5', 'r7', 'd1', 'l3', 'u1')]
    cube[8:, i] = cube[8:, j]


def Lc(cube):
    """Rotates side L clockwise inplace"""
    # Corners
    # -------
    # [l0, l2, l4, l6] <= [l2, l4, l6, l0]
    # [f0, f6] <= [u0, u6]
    # [u0, u6] <= [b4, b2]
    # [b4, b2] <= [d6, d0]
    # [d6, d0] <= [f0, f6]
    #
    # l0 l2 l4 l6 f0 f6 u0 u6 b4 b2 d6 d0
    # l2 l4 l6 l0 u0 u6 b4 b2 d6 d0 f0 f6
    #
    # Edges
    # -----
    # [l1, l3, l5, l7] <= [l3, l5, l7, l1]
    # [f7, u7, b3, d7] <= [u7, b3, d7, f7]
    #
    # l1 l3 l5 l7 f7 u7 b3 d7
    # l3 l5 l7 l1 u7 b3 d7 f7
    cmap = _CORNER_TABLE_COL_INDEX
    emap = _EDGE_TABLE_COL_INDEX

    # Swap corners
    i = [cmap[c] for c in ('l0', 'l2', 'l4', 'l6', 'f0', 'f6', 'u0', 'u6', 'b4', 'b2', 'd6', 'd0')]
    j = [cmap[c] for c in ('l2', 'l4', 'l6', 'l0', 'u0', 'u6', 'b4', 'b2', 'd6', 'd0', 'f0', 'f6')]
    cube[:8, i] = cube[:8, j]

    # Swap edges
    i = [emap[c] for c in ('l1', 'l3', 'l5', 'l7', 'f7', 'u7', 'b3', 'd7')]
    j = [emap[c] for c in ('l3', 'l5', 'l7', 'l1', 'u7', 'b3', 'd7', 'f7')]
    cube[8:, i] = cube[8:, j]


def Lcc(cube):
    """Rotates side L counterclockwise inplace"""
    # Corners
    # -------
    # [l0, l2, l4, l6] <= [l6, l0, l2, l4]
    # [u6, u0] <= [f6, f0]
    # [f6, f0] <= [d0, d6]
    # [d0, d6] <= [b2, b4]
    # [b2, b4] <= [u6, u0]
    #
    # l0 l2 l4 l6 u6 u0 f6 f0 d0 d6 b2 b4
    # l6 l0 l2 l4 f6 f0 d0 d6 b2 b4 u6 u0
    #
    # Edges
    # -----
    # [l1, l3, l5, l7] <= [l7, l1, l3, l5]
    # [f7, u7, b3, d7] <= [d7, f7, u7, b3]
    #
    # l1 l3 l5 l7 f7 u7 b3 d7
    # l7 l1 l3 l5 d7 f7 u7 b3
    cmap = _CORNER_TABLE_COL_INDEX
    emap = _EDGE_TABLE_COL_INDEX

    # Swap corners
    i = [cmap[c] for c in ('l0', 'l2', 'l4', 'l6', 'u6', 'u0', 'f6', 'f0', 'd0', 'd6', 'b2', 'b4')]
    j = [cmap[c] for c in ('l6', 'l0', 'l2', 'l4', 'f6', 'f0', 'd0', 'd6', 'b2', 'b4', 'u6', 'u0')]
    cube[:8, i] = cube[:8, j]

    # Swap edges
    i = [emap[c] for c in ('l1', 'l3', 'l5', 'l7', 'f7', 'u7', 'b3', 'd7')]
    j = [emap[c] for c in ('l7', 'l1', 'l3', 'l5', 'd7', 'f7', 'u7', 'b3')]
    cube[8:, i] = cube[8:, j]


def Rc(cube):
    """Rotates side R clockwise inplace"""
    # Corners
    # -------
    # [r0, r6, r4, r2] <= [r2, r0, r6, r4]
    # [u4, u2] <= [f4, f2]
    # [f4, f2] <= [d2, d4]
    # [d2, d4] <= [b0, b6]
    # [b0, b6] <= [u4, u2]
    #
    # r0 r6 r4 r2 u4 u2 f4 f2 d2 d4 b0 b6
    # r2 r0 r6 r4 f4 f2 d2 d4 b0 b6 u4 u2
    #
    # Edges
    # -----
    # [r1, r7, r5, r3] <= [r3, r1, r7, r5]
    # [f3, u3, b7, d3] <= [d3, f3, u3, b7]
    #
    # r1 r7 r5 r3 f3 u3 b7 d3
    # r3 r1 r7 r5 d3 f3 u3 b7
    cmap = _CORNER_TABLE_COL_INDEX
    emap = _EDGE_TABLE_COL_INDEX

    # Swap corners
    i = [cmap[c] for c in ('r0', 'r6', 'r4', 'r2', 'u4', 'u2', 'f4', 'f2', 'd2', 'd4', 'b0', 'b6')]
    j = [cmap[c] for c in ('r2', 'r0', 'r6', 'r4', 'f4', 'f2', 'd2', 'd4', 'b0', 'b6', 'u4', 'u2')]
    cube[:8, i] = cube[:8, j]

    # Swap edges
    i = [emap[c] for c in ('r1', 'r7', 'r5', 'r3', 'f3', 'u3', 'b7', 'd3')]
    j = [emap[c] for c in ('r3', 'r1', 'r7', 'r5', 'd3', 'f3', 'u3', 'b7')]
    cube[8:, i] = cube[8:, j]


def Rcc(cube):
    """Rotates side R counterclockwise inplace"""
    # Corners
    # -------
    # [r0, r6, r4, r2] <= [r6, r4, r2, r0]
    # [f2, f4] <= [u2, u4]
    # [u2, u4] <= [b6, b0]
    # [b6, b0] <= [d4, d2]
    # [d4, d2] <= [f2, f4]
    #
    # r0 r6 r4 r2 f2 f4 u2 u4 b6 b0 d4 d2
    # r6 r4 r2 r0 u2 u4 b6 b0 d4 d2 f2 f4
    #
    # Edges
    # -----
    # [r1, r7, r5, r3] <= [r7, r5, r3, r1]
    # [f3, u3, b7, d3] <= [u3, b7, d3, f3]
    #
    # r1 r7 r5 r3 f3 u3 b7 d3
    # r7 r5 r3 r1 u3 b7 d3 f3
    cmap = _CORNER_TABLE_COL_INDEX
    emap = _EDGE_TABLE_COL_INDEX

    # Swap corners
    i = [cmap[c] for c in ('r0', 'r6', 'r4', 'r2', 'f2', 'f4', 'u2', 'u4', 'b6', 'b0', 'd4', 'd2')]
    j = [cmap[c] for c in ('r6', 'r4', 'r2', 'r0', 'u2', 'u4', 'b6', 'b0', 'd4', 'd2', 'f2', 'f4')]
    cube[:8, i] = cube[:8, j]

    # Swap edges
    i = [emap[c] for c in ('r1', 'r7', 'r5', 'r3', 'f3', 'u3', 'b7', 'd3')]
    j = [emap[c] for c in ('r7', 'r5', 'r3', 'r1', 'u3', 'b7', 'd3', 'f3')]
    cube[8:, i] = cube[8:, j]


def Dc(cube):
    """Rotates side D clockwise inplace"""
    # Corners
    # -------
    # [d0, d2, d4, d6] <= [d6, d0, d2, d4]
    # [f0, f2] <= [l0, l2]
    # [l0, l2] <= [b0, b2]
    # [b0, b2] <= [r0, r2]
    # [r0, r2] <= [f0, f2]
    #
    # d0 d2 d4 d6 f0 f2 l0 l2 b0 b2 r0 r2
    # d6 d0 d2 d4 l0 l2 b0 b2 r0 r2 f0 f2
    #
    # Edges
    # -----
    # [d1, d3, d5, d7] <= [d7, d1, d3, d5]
    # [f1, r1, b1, l1] <= [l1, f1, r1, b1]
    #
    # d1 d3 d5 d7 f1 r1 b1 l1
    # d7 d1 d3 d5 l1 f1 r1 b1
    cmap = _CORNER_TABLE_COL_INDEX
    emap = _EDGE_TABLE_COL_INDEX

    # Swap corners
    i = [cmap[c] for c in ('d0', 'd2', 'd4', 'd6', 'f0', 'f2', 'l0', 'l2', 'b0', 'b2', 'r0', 'r2')]
    j = [cmap[c] for c in ('d6', 'd0', 'd2', 'd4', 'l0', 'l2', 'b0', 'b2', 'r0', 'r2', 'f0', 'f2')]
    cube[:8, i] = cube[:8, j]

    # Swap edges
    i = [emap[c] for c in ('d1', 'd3', 'd5', 'd7', 'f1', 'r1', 'b1', 'l1')]
    j = [emap[c] for c in ('d7', 'd1', 'd3', 'd5', 'l1', 'f1', 'r1', 'b1')]
    cube[8:, i] = cube[8:, j]


def Dcc(cube):
    """Rotates side D counterclockwise inplace"""
    # Corners
    # -------
    # [d0, d2, d4, d6] <= [d2, d4, d6, d0]
    # [l0, l2] <= [f0, f2]
    # [f0, f2] <= [r0, r2]
    # [r0, r2] <= [b0, b2]
    # [b0, b2] <= [l0, l2]
    #
    # d0 d2 d4 d6 l0 l2 f0 f2 r0 r2 b0 b2
    # d2 d4 d6 d0 f0 f2 r0 r2 b0 b2 l0 l2
    #
    # Edges
    # -----
    # [d1, d3, d5, d7] <= [d3, d5, d7, d1]
    # [l1, f1, r1, b1] <= [f1, r1, b1, l1]
    #
    # d1 d3 d5 d7 l1 f1 r1 b1
    # d3 d5 d7 d1 f1 r1 b1 l1
    cmap = _CORNER_TABLE_COL_INDEX
    emap = _EDGE_TABLE_COL_INDEX

    # Swap corners
    i = [cmap[c] for c in ('d0', 'd2', 'd4', 'd6', 'l0', 'l2', 'f0', 'f2', 'r0', 'r2', 'b0', 'b2')]
    j = [cmap[c] for c in ('d2', 'd4', 'd6', 'd0', 'f0', 'f2', 'r0', 'r2', 'b0', 'b2', 'l0', 'l2')]
    cube[:8, i] = cube[:8, j]

    # Swap edges
    i = [emap[c] for c in ('d1', 'd3', 'd5', 'd7', 'l1', 'f1', 'r1', 'b1')]
    j = [emap[c] for c in ('d3', 'd5', 'd7', 'd1', 'f1', 'r1', 'b1', 'l1')]
    cube[8:, i] = cube[8:, j]


def Bc(cube):
    """Rotates side B clockwise inplace"""
    # Corners
    # -------
    # [b0, b2, b4, b6] <= [b2, b4, b6, b0]
    # [l0, l6] <= [u6, u4]
    # [u6, u4] <= [r4, r2]
    # [r4, r2] <= [d4, d6]
    # [d4, d6] <= [l0, l6]
    #
    # b0 b2 b4 b6 l0 l6 u6 u4 r4 r2 d4 d6
    # b2 b4 b6 b0 u6 u4 r4 r2 d4 d6 l0 l6
    #
    # Edges
    # -----
    # [b1, b3, b5, b7] <= [b3, b5, b7, b1]
    # [l7, u5, r3, d5] <= [u5, r3, d5, l7]
    #
    # b1 b3 b5 b7 l7 u5 r3 d5
    # b3 b5 b7 b1 u5 r3 d5 l7
    cmap = _CORNER_TABLE_COL_INDEX
    emap = _EDGE_TABLE_COL_INDEX

    # Swap corners
    i = [cmap[c] for c in ('b0', 'b2', 'b4', 'b6', 'l0', 'l6', 'u6', 'u4', 'r4', 'r2', 'd4', 'd6')]
    j = [cmap[c] for c in ('b2', 'b4', 'b6', 'b0', 'u6', 'u4', 'r4', 'r2', 'd4', 'd6', 'l0', 'l6')]
    cube[:8, i] = cube[:8, j]

    # Swap edges
    i = [emap[c] for c in ('b1', 'b3', 'b5', 'b7', 'l7', 'u5', 'r3', 'd5')]
    j = [emap[c] for c in ('b3', 'b5', 'b7', 'b1', 'u5', 'r3', 'd5', 'l7')]
    cube[8:, i] = cube[8:, j]


def Bcc(cube):
    """Rotates side B counterclockwise inplace"""
    # Corners
    # -------
    # [b0, b2, b4, b6] <= [b6, b0, b2, b4]
    # [u6, u4] <= [l0, l6]
    # [l0, l6] <= [d4, d6]
    # [d4, d6] <= [r4, r2]
    # [r4, r2] <= [u6, u4]
    #
    # b0 b2 b4 b6 u6 u4 l0 l6 d4 d6 r4 r2
    # b6 b0 b2 b4 l0 l6 d4 d6 r4 r2 u6 u4
    #
    # Edges
    # -----
    # [b1, b3, b5, b7] <= [b7, b1, b3, b5]
    # [u5, l7, d5, r3] <= [l7, d5, r3, u5]
    #
    # b1 b3 b5 b7 u5 l7 d5 r3
    # b7 b1 b3 b5 l7 d5 r3 u5
    cmap = _CORNER_TABLE_COL_INDEX
    emap = _EDGE_TABLE_COL_INDEX

    # Swap corners
    i = [cmap[c] for c in ('b0', 'b2', 'b4', 'b6', 'u6', 'u4', 'l0', 'l6', 'd4', 'd6', 'r4', 'r2')]
    j = [cmap[c] for c in ('b6', 'b0', 'b2', 'b4', 'l0', 'l6', 'd4', 'd6', 'r4', 'r2', 'u6', 'u4')]
    cube[:8, i] = cube[:8, j]

    # Swap edges
    i = [emap[c] for c in ('b1', 'b3', 'b5', 'b7', 'u5', 'l7', 'd5', 'r3')]
    j = [emap[c] for c in ('b7', 'b1', 'b3', 'b5', 'l7', 'd5', 'r3', 'u5')]
    cube[8:, i] = cube[8:, j]


def pure(action):
    """
    Constrcts and returns pure action from impure (modifying) action
    """
    @wraps(action)
    def f(cube):
        state = cube.copy()
        action(state)
        return state
    return f


ACTIONS = tuple(pure(a) for a in (Uc, Ucc, Dc, Dcc, Lc, Lcc,
                                  Rc, Rcc, Fc, Fcc, Bc, Bcc))

# action_map = dict(enumerate(actions))


# TODO Refactor
def test():
    cube = solved_cube()
    # U
    Ac, Acc = pure(Uc), pure(Ucc)
    assert(np.any(Ac(cube) != cube))
    assert(np.any(Acc(cube) != cube))
    assert(np.all(Acc(Ac(cube)) == cube))
    assert(np.all(Ac(Acc(cube)) == cube))
    # D
    Ac, Acc = pure(Dc), pure(Dcc)
    assert(np.any(Ac(cube) != cube))
    assert(np.any(Acc(cube) != cube))
    assert(np.all(Acc(Ac(cube)) == cube))
    assert(np.all(Ac(Acc(cube)) == cube))
    # L
    Ac, Acc = pure(Lc), pure(Lcc)
    assert(np.any(Ac(cube) != cube))
    assert(np.any(Acc(cube) != cube))
    assert(np.all(Acc(Ac(cube)) == cube))
    assert(np.all(Ac(Acc(cube)) == cube))
    # R
    Ac, Acc = pure(Rc), pure(Rcc)
    assert(np.any(Ac(cube) != cube))
    assert(np.any(Acc(cube) != cube))
    assert(np.all(Acc(Ac(cube)) == cube))
    assert(np.all(Ac(Acc(cube)) == cube))
    # F
    Ac, Acc = pure(Fc), pure(Fcc)
    assert(np.any(Ac(cube) != cube))
    assert(np.any(Acc(cube) != cube))
    assert(np.all(Acc(Ac(cube)) == cube))
    assert(np.all(Ac(Acc(cube)) == cube))
    # B
    Ac, Acc = pure(Bc), pure(Bcc)
    assert(np.any(Ac(cube) != cube))
    assert(np.any(Acc(cube) != cube))
    assert(np.all(Acc(Ac(cube)) == cube))
    assert(np.all(Ac(Acc(cube)) == cube))

    np.random.seed(7)
    forward = (Uc, Dc, Lc, Rc, Fc, Bc)
    backward = (Ucc, Dcc, Lcc, Rcc, Fcc, Bcc)
    for _ in range(1000):
        a = np.random.randint(0, 6, 50)
        cube = solved_cube()
        for i in a:
            forward[i](cube)
        for i in reversed(a):
            backward[i](cube)
        assert(np.all(cube == solved_cube()))

    forward = ('Uc', 'Dc', 'Lc', 'Rc', 'Fc', 'Bc')
    backward = ('Ucc', 'Dcc', 'Lcc', 'Rcc', 'Fcc', 'Bcc')
    all_actions = forward + backward

    G = globals()
    action_func_map = {n: G[n] for n in all_actions}
    print(action_func_map)
    reverse_action_map = dict(zip(forward, backward))
    reverse_action_map.update(zip(backward, forward))
    print(reverse_action_map)

    np.random.seed(17)
    for _ in range(1000):
        a = np.random.randint(0, 12, 100)
        cube = solved_cube()
        taken = [all_actions[i] for i in a]
        for t in taken:
            action_func_map[t](cube)
        rev = [reverse_action_map[t] for t in taken]
        for t in reversed(rev):
            action_func_map[t](cube)
        assert(np.all(cube == solved_cube()))


# test()


class RubickCube:
    """Representation as 24 x 20 matrix """

    terminal_state = solved_cube()
    _actions = (Uc, Ucc, Dc, Dcc, Lc, Lcc,
                Rc, Rcc, Fc, Fcc, Bc, Bcc)

    def __init__(self):
        self.state = solved_cube()

    def reset(self):
        """ Reset to solved state """
        self.state = RubickCube.terminal_state.copy()

    @classmethod
    def is_terminal(cls, state):
        """
        Return True if the state is the terminal state for the environment.

        Parameters
        ----------
            state : Numpy array with shape (20, 24).

        Returns
        -------
            True if the state is terminal for the environment.
        """

    @classmethod
    def reward(cls, state):
        """
        Return the immediate reward on transition to state `state`

        Parameters
        ----------
            state : Numpy array with shape (20, 24)
        """
        return 1 if cls.is_terminal(state) else -1

    @classmethod
    def expand_state(cls, state):
        """
        Return all 12 possible next states from `state`

        Parameters
        ----------
            state : Numpy array with shape (20, 24)

        Returns
        -------
            (children, rewards) tuple.

            children : Numpy array with shape (12, 20, 24)
            rewards : Numpy array with shape (12, 1)
        """
        children = np.array([ACTIONS[a](state) for a in range(12)])
        solved = RubickCube.terminal_state
        rewards = np.array([1 if np.all(s == solved) else -1 for s in children])
        return children, rewards

    def step(self, a):
        """ Make a move inplace. Return (state, reward, done) tuple """
        RubickCube._actions[a](self.state)
        done = self.is_solved()
        reward = 1 if done else -1
        return self.state.copy(), reward, done

    def move(self, a):
        """
        Makes a move and returns (state, reward, done) tuple.
        Does not modifies `self.state`.
        """
        s = self.state.copy()
        state, reward, done = self.step(a)
        self.state = s
        return state, reward, done

    def set_random_state(self):
        """ Make 100 scrambles, starting from solved state """
        self.reset()
        step = self.step
        for _ in range(100):
            step(np.random.randint(0, 12))

    def is_solved(self):
        """ Returns true if `self.state` is the terminal state """
        return np.all(self.state == RubickCube.terminal_state)

    def shuffle(self, n):
        """Make `n` random moves to `self`. """
        for m in np.random.randint(0, 12, n):
            self.step(m)


def expand_state(state: np.ndarray):
    """ Returns all 12 children of `cube` in a tuple """
    children = [ACTIONS[a](state) for a in range(12)]
    solved = solved_cube()
    rewards = [1 if np.all(s == solved) else -1 for s in children]
    return children, rewards


def expand_states(states):
    """ states : List of Numpy ndarrays """
    children, rewards = list(zip(*map(expand_state, states)))
    children = np.vstack(children)
    rewards = np.vstack(rewards).reshape(-1, 1)
    children = np.expand_dims(children, 3).astype(np.float)
    return children, rewards

