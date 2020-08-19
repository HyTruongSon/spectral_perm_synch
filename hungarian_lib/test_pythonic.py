import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np

import networkx as nx

from hungarian_pythonic_lib import bipartite_matching

# Batch size
B = 10

# Number of vertices
N = 20

# Number of channels
C = 16

def permute(tensor1, tensor2):
	B = tensor1.size(0)
	N = tensor1.size(1)
	C = tensor1.size(2)
	result1 = torch.zeros(tensor1.size())
	result2 = torch.zeros(tensor2.size())
	for b in range(B):
		for c in range(C):
			perm = np.random.permutation(N)
			for n in range(N):
				result1[b, n, c] = tensor1[b, perm[n], c]
				result2[b, n, c] = tensor2[b, perm[n], c]
	return result1, result2

mu_encoder = torch.randn(B, N, C)
logvar_encoder = torch.randn(B, N, C)
mu_prior, logvar_prior = permute(mu_encoder, logvar_encoder)

perm = bipartite_matching(mu_encoder, mu_prior, logvar_encoder, logvar_prior)
assert torch.sum(torch.einsum('bic,bijc->bjc', mu_encoder, perm) - mu_prior).item() == 0
assert torch.sum(torch.einsum('bic,bijc->bjc', logvar_encoder, perm) - logvar_prior).item() == 0
print("Done testing pythonic")