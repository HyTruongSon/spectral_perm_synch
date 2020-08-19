import numpy as np
import torch

import hungarian_lib

# Matching by features
B = 10 # Number of bipartite graphs
N = 20 # Number of vertices in each side of a bipartite graph
C = 3 # Number of features associated with each vertex
nThreads = B # Number of threads

feature_x = torch.randn(B, N, C)
feature_y = torch.randn(B, N, C)
perm = torch.zeros(B, N, N)
hungarian_lib.bipartite_matching(feature_x, feature_y, perm, nThreads, 0)
print("Done matching given features")


# Matching by cost matrix
B = 10 # Number of bipartite graphs
N = 20 # Number of vertices in each side of a bipartite graph
nThreads = B # Number of threads

cost = torch.randn(B, N, N)
cost = cost - torch.min(cost)
perm = torch.zeros(B, N, N)
hungarian_lib.bipartite_matching_3(cost, perm, nThreads)
print("Done matching given cost matrix")