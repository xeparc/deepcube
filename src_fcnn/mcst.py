"""
Implements Monte Carlo Search Tree
"""
import numpy as np
from collections import deque


# Stub function
def expand_children(node):
    children = [0] * 12
    state = node.state
    for i in range(12):
        child_state = state.copy()
        np.random.shuffle(child_state)
        children[i] = Node(child_state)
    result = Node(state)
    result.children = children
    return result


# Stub
def get_children(state):
    child = state.copy()
    children = [0] * 12
    for i in range(12):
        np.random.shuffle(child)
        children[i] = child.copy()
    return children

# Stub
def is_solved(state):
    return np.random.randint(0, 1000) < 10


class Node:

    __slots__ = (
        '__weakref__',
        'id',
        'state',
        'children',
        'priors',
        'weights',
        'visits',
        'losses'
    )

    ExpandCallback = expand_children
    C_param = 1.0
    VirtualLoss = 0.0
    Counts = 1

    def __init__(self, state):
        """
        Description
        -----------
        Initialize MCST node from game state
        """
        # Attributes applicable to `self`
        self.state = state
        self.id = Node.Counts
        Node.Counts += 1
        # Attributes applicable to `self`'s children (or edges)
        self.children = []
        self.priors = np.zeros(12, dtype=np.float32)
        self.weights = np.zeros(12, dtype=np.float32)
        self.visits = np.zeros(12, dtype=np.int)
        self.losses = np.zeros(12, dtype=np.float32)

    def __hash__(self):
        return self.id

    def eval_tree_policy(self):
        """
        Description
        -----------
        Evaluates the Tree Policy At = argmax Ust(a) + Qst(a) for each child
        of `self`

        Returns
        -------
        Index in `self.children` that points to the child maximizing the Tree
        Policy
        """
        total_visits = (np.sum(self.visits)) ** 0.5
        U = Node.C_param * self.priors * (total_visits / (1 + self.visits))
        Q = self.weights - self.losses
        return np.argmax(U + Q)

    def expand(self):
        """
        Description
        -----------
        Expands all unvisited leaf nodes with 1 level BFS.
        Modifies the tree in-place

        Returns
        -------
        `None`
        """
        Q = deque([self])
        while Q:
            front = Q.popleft()
            children = front.children
            if not children:
                # The proper initialization of prior probabilities for
                # the actions is ignored
                front.children = [Node(c) for c in get_children(front.state)]
            else:
                Q.extend(children)

    def find_solution(self):
        """
        Description
        -----------
        Returns the shortest path to Terminal State as list of actions
        """
        Q = deque([self])
        # node ID => (parent node, action)
        parents = {self.id: (None, 0)}
        while Q:
            front = Q.popleft()
            if is_solved(front.state):
                break
            children = front.children
            for a, c in enumerate(children):
                assert(c.id not in parents)
                parents[c.id] = (front, a)
            Q.extend(children)
        else:
            return []
        # Construct the path from `front` to Tree's root
        it = front
        path = []
        while it:
            it, a = parents[front.id]
            path.append(a)
        return path[::-1]


def simulate(root, approximator):
    """
    Description
    -----------
    Simulates single tree traversal guided by `approximator`.
    Updates the tree inplace.

    Returns
    -------
    `True` if the terminal state can be reached from some leaf, otherwise `False`
    """
    nodes = []
    actions = []
    # Simulate until reaching leaf node
    node = root
    tree_policy = root.eval_tree_policy
    while node.children:
        nodes.append(node)
        best = tree_policy(node)
        actions.append(best)
        node = node.children[best]
    # Leaf node reached. Expand the state by adding all children
    children = get_children(node.state)
    # Check if any of the children is the Terminal State
    if any(is_solved(c) for c in children):
        return True
    policy, V = approximator.evaluate(node.state)
    # Update prior probabilities of node's children
    node.priors = policy
    # Update the chosen actions counts and virtual loss parameters
    vloss = Node.VirtualLoss
    for s, a in zip(nodes, actions):
        s.visits[a] += 1
        s.losses[a] -= vloss
    # Backpropagate `V`
    for n, a in zip(nodes, actions):
        n.weights[a] = max(V, n.weights[a])
    return False


def solve(state, approximator, max_iterations=10_000):
    """
    Description
    -----------
    Solves the game with MCTS guided by `approximator`

    Parameters
    ----------
    state :
        Starting state
    approximator :
        Function approximator for the policy and value of a game state
    max_iterations :
        Maximum number of MCTS simulations

    Returns
    -------
    List of moves that solve the game. If no terminal state is encountered
    for `max_iterations` traversals, then empty list is returned.
    """
    root = Node(state)
    for _ in range(max_iterations):
        reached = simulate(root, approximator)
        if reached:
            root.expand()
            moves = root.find_solution()
            assert(moves)
            return moves
    return []
