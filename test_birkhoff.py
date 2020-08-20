import numpy as np
import torch

import hungarian_lib

def Birkhoff_algorithm(X):
	N = X.shape[0]
	z = []
	P = []
	while np.sum(np.abs(X)) > 1e-6:
		perm = torch.zeros(1, N, N).type(torch.FloatTensor)
		cost = torch.from_numpy(X).reshape(1, N, N).type(torch.FloatTensor)
		cost = torch.max(cost) - cost
		hungarian_lib.bipartite_matching_3(cost, perm, 1)
		perm = np.reshape(perm.numpy(), (N, N))
		smallest = np.min(X[perm == 1])
		z.append(smallest)
		P.append(perm)
		X -= smallest * perm
	return z, P

X_in = np.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2], [0.2, 0.5, 0.3]])
X = np.array(X_in)
N = X_in.shape[0]

assert np.sum(np.abs(np.ones((N, 1)) - np.matmul(X_in, np.ones((N, 1))))) == 0
assert np.sum(np.abs(np.ones((1, N)) - np.matmul(np.ones((1, N)), X_in))) == 0

z, P = Birkhoff_algorithm(X_in)

X_hat = np.zeros((N, N))
for i in range(len(z)):
	X_hat += z[i] * P[i]

diff = np.sum(np.abs(X - X_hat))
assert diff < 1e-6

print(z)
print(P)
print("Done")