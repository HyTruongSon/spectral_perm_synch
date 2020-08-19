import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np

import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.bipartite import matching
from scipy.sparse import csr_matrix

import collections
import itertools

from networkx.algorithms.bipartite.matrix import biadjacency_matrix
from networkx.algorithms.bipartite import sets as bipartite_sets

def minimum_weight_full_matching(G, top_nodes=None, weight='weight'):
    r"""Returns the minimum weight full matching of the bipartite graph `G`.

    Let :math:`G = ((U, V), E)` be a complete weighted bipartite graph with
    real weights :math:`w : E \to \mathbb{R}`. This function then produces
    a maximum matching :math:`M \subseteq E` which, since the graph is
    assumed to be complete, has cardinality
   
    .. math::
       \lvert M \rvert = \min(\lvert U \rvert, \lvert V \rvert),

    and which minimizes the sum of the weights of the edges included in the
    matching, :math:`\sum_{e \in M} w(e)`.
    
    When :math:`\lvert U \rvert = \lvert V \rvert`, this is commonly
    referred to as a perfect matching; here, since we allow
    :math:`\lvert U \rvert` and :math:`\lvert V \rvert` to differ, we
    follow Karp [1]_ and refer to the matching as *full*.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed.

    weight : string, optional (default='weight')

       The edge data key used to provide each value in the matrix.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matches`, such that
      ``matches[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in matches.

    Raises
    ------
    ValueError : Exception

      Raised if the input bipartite graph is not complete.

    ImportError : Exception

      Raised if SciPy is not available.

    Notes
    -----
    The problem of determining a minimum weight full matching is also known as
    the rectangular linear assignment problem. This implementation defers the
    calculation of the assignment to SciPy.

    References
    ----------
    .. [1] Richard Manning Karp:
       An algorithm to Solve the m x n Assignment Problem in Expected Time
       O(mn log n).
       Networks, 10(2):143–152, 1980.

    """
    try:
        import scipy.optimize
    except ImportError:
        raise ImportError('minimum_weight_full_matching requires SciPy: ' +
                          'https://scipy.org/')
    left, right = nx.bipartite.sets(G, top_nodes)
    # Ensure that the graph is complete. This is currently a requirement in
    # the underlying  optimization algorithm from SciPy, but the constraint
    # will be removed in SciPy 1.4.0, at which point it can also be removed
    # here.
    '''
    for (u, v) in itertools.product(left, right):
        # As the graph is undirected, make sure to check for edges in
        # both directions
        if (u, v) not in G.edges():# and (v, u) not in G.edges():
            raise ValueError('The bipartite graph must be complete.')
    '''
    U = list(left)
    V = list(right)
    weights = biadjacency_matrix(G, row_order=U,
                                 column_order=V, weight=weight).toarray()
    left_matches = scipy.optimize.linear_sum_assignment(weights)
    d = {U[u]: V[v] for u, v in zip(*left_matches)}
    # d will contain the matching from edges in left to right; we need to
    # add the ones from right to left as well.
    d.update({v: u for u, v in d.items()})
    return d

def KL_divergence(mu1, logvar1, mu2, logvar2):
  sigma1 = np.exp(logvar1)
  sigma2 = np.exp(logvar2)
  return logvar2 - logvar1 - 0.5 + 0.5 * (sigma1**2 + (mu1 - mu2)**2) / (sigma2**2)

def bipartite_matching(mu_encoder, mu_prior, logvar_encoder, logvar_prior):
  B = mu_encoder.size(0)
  N = mu_encoder.size(1)
  C = mu_encoder.size(2)
  perm = torch.zeros(B, N, N, C)
  for b in range(B):
    for c in range(C):
      cost = np.zeros((N, N))
      for i in range(N):
        for j in range(N):
          cost[i, j] = KL_divergence(mu_encoder[b, i, c], logvar_encoder[b, i, c], mu_prior[b, j, c], logvar_prior[b, j, c])
      G = bipartite.from_biadjacency_matrix(csr_matrix(cost))
      match = minimum_weight_full_matching(G)
      for i in range(N):
        j = match[i]
        perm[b, i, j - N, c] = 1
  return perm

def free_matching(mu_encoder, mu_prior, logvar_encoder, logvar_prior):
  B = mu_encoder.size(0)
  N = mu_encoder.size(1)
  C = mu_encoder.size(2)
  perm = torch.zeros(B, N, N, C)
  for b in range(B):
    for c in range(C):
      cost = np.zeros((N, N))
      for i in range(N):
        for j in range(N):
          cost[i, j] = KL_divergence(mu_encoder[b, i, c], logvar_encoder[b, i, c], mu_prior[b, j, c], logvar_prior[b, j, c])
        index = np.argmin(cost[i, :])
        perm[b, i, index, c] = 1
  return perm
