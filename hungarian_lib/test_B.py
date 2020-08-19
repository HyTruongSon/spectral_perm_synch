# Test
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np

import hungarian_lib

# Batch size
B = 20

# Number of vertices
N = 40

# Number of channels
C = 32

# Separate channels
separate_channels = 0

def permute(tensor):
	B = tensor.size(0)
	N = tensor.size(1)
	C = tensor.size(2)
	result = torch.zeros(tensor.size())
	for b in range(B):
		perm = np.random.permutation(N)
		for c in range(C):
			for n in range(N):
				result[b, n, c] = tensor[b, perm[n], c]
	return result

def permute_2(tensor1, tensor2):
	B = tensor1.size(0)
	N = tensor1.size(1)
	C = tensor1.size(2)
	result1 = torch.zeros(tensor1.size())
	result2 = torch.zeros(tensor2.size())
	for b in range(B):
		perm = np.random.permutation(N)
		for c in range(C):
			for n in range(N):
				result1[b, n, c] = tensor1[b, perm[n], c]
				result2[b, n, c] = tensor2[b, perm[n], c]
	return result1, result2

def permute_full_sigma(tensor1, tensor2):
	B = tensor1.size(0)
	N = tensor1.size(1)
	C = tensor1.size(2)
	result1 = torch.zeros(tensor1.size())
	result2 = torch.zeros(tensor2.size())
	for b in range(B):
		perm = np.random.permutation(N)
		for c in range(C):
			for n in range(N):
				result1[b, n, c] = tensor1[b, perm[n], c]
				for n_ in range(N):
					result2[b, n, n_, c] = tensor2[b, perm[n], perm[n_], c]
	return result1, result2

for iter in range(10):
	print("Iteration " + str(iter))

	# Test 1
	mu_encoder = torch.randn(B, N, C)
	mu_prior = mu_encoder
	perm = torch.zeros(B, N, N)
	hungarian_lib.bipartite_matching(mu_encoder, mu_prior, perm, 20, separate_channels)
	for b in range(B):
		assert torch.sum(perm[b, :, :] - torch.eye(N)).item() == 0
	print("Done test 1")

	# Test 2
	mu_encoder = torch.randn(B, N, C)
	mu_prior = permute(mu_encoder)
	perm = torch.zeros(B, N, N)
	hungarian_lib.bipartite_matching(mu_encoder, mu_prior, perm, 20, separate_channels)
	assert torch.sum(torch.einsum('bic,bij->bjc', mu_encoder, perm) - mu_prior).item() == 0
	print("Done test 2")

	# Test 3
	mu_encoder = torch.randn(B, N, C)
	mu_prior = mu_encoder
	sigma_encoder = torch.randn(B, N, C)
	sigma_prior = sigma_encoder
	perm = torch.zeros(B, N, N)
	hungarian_lib.bipartite_matching_2(mu_encoder, mu_prior, sigma_encoder, sigma_prior, perm, 20, separate_channels)
	for b in range(B):
		assert torch.sum(perm[b, :, :] - torch.eye(N)).item() == 0
	print("Done test 3")

	# Test 6
	mu_encoder = torch.randn(B, N, C)
	sigma_encoder = torch.randn(B, N, C)
	mu_prior, sigma_prior = permute_2(mu_encoder, sigma_encoder)
	perm = torch.zeros(B, N, N)
	hungarian_lib.bipartite_matching_2(mu_encoder, mu_prior, sigma_encoder, sigma_prior, perm, 20, separate_channels)
	assert torch.sum(torch.einsum('bic,bij->bjc', mu_encoder, perm) - mu_prior).item() == 0
	assert torch.sum(torch.einsum('bic,bij->bjc', sigma_encoder, perm) - sigma_prior).item() == 0
	print("Done test 4")

# Test 9 - Full sigma
mu_encoder = torch.randn(B, N, C)
sigma_encoder = torch.randn(B, N, N, C)
mu_prior, sigma_prior = permute_full_sigma(mu_encoder, sigma_encoder)
diag_encoder = torch.diagonal(sigma_encoder, dim1 = 1, dim2 = 2).transpose(1, 2)
diag_prior = torch.diagonal(sigma_prior, dim1 = 1, dim2 = 2).transpose(1, 2)
perm = torch.zeros(B, N, N)
hungarian_lib.bipartite_matching(mu_encoder, mu_prior, perm, 20, separate_channels)
assert torch.sum(torch.einsum('bic,bij->bjc', mu_encoder, perm) - mu_prior).item() == 0
part = torch.einsum('bijc,bjk->bikc', sigma_encoder, perm)
full = torch.einsum('bji,bikc->bjkc', perm.transpose(1, 2), part)
assert torch.sum(full - sigma_prior).item() == 0
print("Done test 5")
