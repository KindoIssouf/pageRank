"""
In this problem, we will implement the complete PageRank algorithm, which is
described on page 44 of the link-analysis slides, and run a simulation using
the example graph shown on the next page. If you see the same values as in the
example nodes, then consider your implementation as correct.
You will write your codes where TODO notes are.
"""
import numpy as np
import numpy.linalg as la

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.set_printoptions(precision=3)
    N = 11  # Number of nodes
    iter_max_n = 100  # Maximum number of iterations
    epsilon = 1e-6  # Itermation termination criteria
    beta = 0.85  # PageRank parameter (DO NOT CHANGE THIS)

    # TODO, Initialize r_old and r_new
    r_old = np.array([1 / N for _ in range(N)])
    r_new = np.array([0 for _ in range(N)])

    # TODO. Write the transition matrix M according to the given example graph
    M = np.zeros((N, N))

    outLinks = {"A": [],
                "B": ["C"],
                "C": ["B"],
                "D": ["A", "B"],
                "E": ["D", "F", "B"],
                "F": ["B", "E"],
                "G": ["B", "E"],
                "H": ["B", "E"],
                "I": ["B", "E"],
                "J": ["E"],
                "K": ["E"]}

    # Matrix initialization
    for i, node in enumerate('ABCDEFGHIJK'):
        for j, otherNode in enumerate('ABCDEFGHIJK'):
            if len(outLinks[otherNode]) > 0 and node in outLinks[otherNode]:
                M[i][j] = 1 / len(outLinks[otherNode])

    for i in range(iter_max_n):
        # TODO. Complete this part that updates r_new as described in the algorithm
        # r_new = (beta * (M@r_old)) + (1-beta) / N
        # print(list(np.dot(M,r_old)))
        r_new_1 = beta * M@r_old
        r_new = r_new_1 + ((1 - np.sum(r_new_1)) / N)

        # TODO. Terminate if the condition meets the criteria
        # That is, use the 2-norm of the difference between r_new and r_old
        if np.linalg.norm(r_new - r_old) < epsilon:
            break
        r_old = r_new
    print(r_old)
    for node, rank in zip('ABCDEFGHIJK', r_new):
        print('node {}: {:.1f}%'.format(node, rank * 100))

"""
Expected printouts are:
node A: 3.3%
node B: 38.4%
node C: 34.3%
node D: 3.9%
node E: 8.1%
node F: 3.9%
node G: 1.6%
node H: 1.6%
node I: 1.6%
node J: 1.6%
node K: 1.6%
"""
