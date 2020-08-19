#include <iostream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <assert.h>

#include <torch/torch.h>

using namespace std;

static int Index(
	const int x1, 
	const int x2, 
	const int X2) 
{
	assert(x2 < X2);
	return x1 * X2 + x2;
}

static int Index(
	const int x1, 
	const int x2, 
	const int x3, 
	const int X2, 
	const int X3) 
{
	assert(x2 < X2);
	assert(x3 < X3);
	return (x1 * X2 + x2) * X3 + x3;
}

static int Index(
	const int x1, 
	const int x2, 
	const int x3, 
	const int x4, 
	const int X2, 
	const int X3, 
	const int X4) 
{
	assert(x2 < X2);
	assert(x3 < X3);
	assert(x4 < X4);
	return ((x1 * X2 + x2) * X3 + x3) * X4 + x4;
}

// +----------------+
// | L1 cost matrix |
// +----------------+

static float L1_distance(
	const int B, 
	const int N, 
	const int C, 
	const int b, 
	const int n1, 
	const int n2, 
	const float *encoder_flat, 
	const float *prior_flat) 
{
	float total = 0.0;
	for (int c = 0; c < C; ++c) {
		const int index1 = Index(b, n1, c, N, C);
		const int index2 = Index(b, n2, c, N, C);
		total += abs(encoder_flat[index1] - prior_flat[index2]);
	}
	return total;
}

static std::vector<std::vector<float>> L1_cost_matrix(
	const int B, 
	const int N, 
	const int C, 
	const int b, 
	const float *mu_encoder_flat, 
	const float *mu_prior_flat) 
{
	std::vector<std::vector<float>> cost;
	for (int n1 = 0; n1 < N; ++n1) {
		std::vector<float> row;
		row.clear();
		for (int n2 = 0; n2 < N; ++n2) {
			const float dist = L1_distance(B, N, C, b, n1, n2, mu_encoder_flat, mu_prior_flat);
			row.push_back(dist);
		}
		cost.push_back(row);
	}
	return cost;
}

static std::vector<std::vector<float>> L1_cost_matrix(
	const int B, 
	const int N, 
	const int C, 
	const int b, 
	const float *mu_encoder_flat, 
	const float *mu_prior_flat, 
	const float *logvar_encoder_flat, 
	const float *logvar_prior_flat) 
{
	std::vector<std::vector<float>> cost;
	for (int n1 = 0; n1 < N; ++n1) {
		std::vector<float> row;
		row.clear();
		for (int n2 = 0; n2 < N; ++n2) {
			const float mu_dist = L1_distance(B, N, C, b, n1, n2, mu_encoder_flat, mu_prior_flat);
			const float logvar_dist = L1_distance(B, N, C, b, n1, n2, logvar_encoder_flat, logvar_prior_flat);
			row.push_back(mu_dist + logvar_dist);
		}
		cost.push_back(row);
	}
	return cost;
}

static std::vector<std::vector<float>> L1_cost_matrix(
	const int B, 
	const int N, 
	const int C, 
	const int b, 
	const int c,
	const float *mu_encoder_flat, 
	const float *mu_prior_flat) 
{
	std::vector<std::vector<float>> cost;
	for (int n1 = 0; n1 < N; ++n1) {
		const int index1 = Index(b, n1, c, N, C);
		const float mu1 = mu_encoder_flat[index1];

		std::vector<float> row;
		row.clear();
		for (int n2 = 0; n2 < N; ++n2) {
			const int index2 = Index(b, n2, c, N, C);
			const float mu2 = mu_prior_flat[index2];

			const float dist = abs(mu1 - mu2);
			row.push_back(dist);
		}
		cost.push_back(row);
	}
	return cost;
}

// +---------------------------+
// | KL-divergence cost matrix |
// +---------------------------+

static float KL_divergence(const float mu1, const float logvar1, const float mu2, const float logvar2) {
	const float sigma1 = exp(logvar1);
	const float sigma2 = exp(logvar2);
	return (logvar2 - logvar1) - 0.5 + (sigma1 * sigma1 + (mu1 - mu2) * (mu1 - mu2)) / (2 * sigma2 * sigma2);
}

static std::vector<std::vector<float>> KL_cost_matrix(
	const int B, 
	const int N, 
	const int C, 
	const int b, 
	const int c,
	const float *mu_encoder_flat, 
	const float *mu_prior_flat,
	const float *logvar_encoder_flat,
	const float *logvar_prior_flat) 
{
	std::vector<std::vector<float>> cost;
	for (int n1 = 0; n1 < N; ++n1) {
		const int index1 = Index(b, n1, c, N, C);
		const float mu1 = mu_encoder_flat[index1];
		const float logvar1 = logvar_encoder_flat[index1];

		std::vector<float> row;
		row.clear();
		for (int n2 = 0; n2 < N; ++n2) {
			const int index2 = Index(b, n2, c, N, C);
			const float mu2 = mu_prior_flat[index2];
			const float logvar2 = logvar_prior_flat[index2];

			const float dist = abs(KL_divergence(mu1, logvar1, mu2, logvar2));
			row.push_back(dist);
		}
		cost.push_back(row);
	}
	return cost;
}

static std::vector<std::vector<float>> KL_cost_matrix(
	const int B, 
	const int N, 
	const int C, 
	const int b, 
	const float *mu_encoder_flat, 
	const float *mu_prior_flat,
	const float *logvar_encoder_flat,
	const float *logvar_prior_flat) 
{
	std::vector<std::vector<float>> cost;
	for (int n1 = 0; n1 < N; ++n1) {
		std::vector<float> row;
		row.clear();
		for (int n2 = 0; n2 < N; ++n2) {
			float dist = 0;
			for (int c = 0; c < C; ++c) {
				const int index1 = Index(b, n1, c, N, C);
				const float mu1 = mu_encoder_flat[index1];
				const float logvar1 = logvar_encoder_flat[index1];

				const int index2 = Index(b, n2, c, N, C);
				const float mu2 = mu_prior_flat[index2];
				const float logvar2 = logvar_prior_flat[index2];

				dist += abs(KL_divergence(mu1, logvar1, mu2, logvar2));
			}
			row.push_back(dist);
		}
		cost.push_back(row);
	}
	return cost;
}

// +--------------------------------------+
// | Bellman-Ford shortest path algorithm |
// +--------------------------------------+

static bool Bellman_Ford(
	const int N, 
	const int source, 
	const int target, 
	float **cost, 
	int **capacity, 
	int **flow, 
	int *queue, 
	bool *in_queue,
	float *d, 
	int *trace)
{
	const int INF = 1e9;
	for (int v = 0; v < N; ++v) {
		d[v] = INF;
		in_queue[v] = false;
		trace[v] = -1;
	}

	int rear = 0;
	int front = 0;

	d[source] = 0;
	queue[front] = source;
	front = (front + 1) % N;
	in_queue[source] = true;

	while (rear != front) {
		const int u = queue[rear];
		rear = (rear + 1) % N;
		in_queue[u] = false;
		for (int v = 0; v < N; ++v) {
			if (capacity[u][v] > flow[u][v]) {
				float dist = d[u];
				if (flow[u][v] >= 0) {
					dist += cost[u][v];
				} else {
					// dist -= cost[v][u];
				}
				if (d[v] > dist) {
					d[v] = dist;
					trace[v] = u;
					if (!in_queue[v]) {
						queue[front] = v;
						front = (front + 1) % N;
						in_queue[v] = true;
					}
				}
			}
		}
	}

	if ((trace[target] == -1) || (d[target] == INF)) {
		return false;
	}
	return true;
}

// +-------------------------------------+
// | Maximum Flow Minimum Cost algorithm |
// +-------------------------------------+

static void increase_flow(const int source, const int target, int **capacity, int **flow, int *trace) {
	int delta = 1e9;
	int v = target;
	while (v != source) {
		const int u = trace[v];
		delta = min(delta, capacity[u][v] - flow[u][v]);
		v = u;
	}
	v = target;
	while (v != source) {
		const int u = trace[v];
		flow[u][v] += delta;
		flow[v][u] -= delta;
		v = u;
	}
}

static std::vector<int> Maximum_Flow_Minimum_Cost(const std::vector<std::vector<float>> &cost_) {
	const int N1 = cost_.size();
	const int N2 = cost_[0].size();
	assert(N1 <= N2);
	const int N = N1 + N2 + 2;
	const int source = N1 + N2;
	const int target = N1 + N2 + 1;

	float **cost = new float* [N];
	int **capacity = new int* [N];
	int **flow = new int* [N];

	for (int u = 0; u < N; ++u) {
		cost[u] = new float [N];
		capacity[u] = new int [N];
		flow[u] = new int [N];

		for (int v = 0; v < N; ++v) {
			cost[u][v] = 0;
			capacity[u][v] = 0;
			flow[u][v] = 0;
		}
	}

	for (int v = 0; v < N1; ++v) {
		capacity[source][v] = 1;
	}
	for (int v = 0; v < N2; ++v) {
		capacity[N1 + v][target] = 1;
	}
	for (int u = 0; u < N1; ++u) {
		for (int v = 0; v < N2; ++v) {
			capacity[u][N1 + v] = 1;
			cost[u][N1 + v] = cost_[u][v];
		}
	}

	int *queue = new int [N];
	bool *in_queue = new bool [N];
	float *d = new float [N];
	int *trace = new int [N];
	
	while (true) {
		bool found = Bellman_Ford(N, source, target, cost, capacity, flow, queue, in_queue, d, trace);
		if (!found) {
			break;
		}
		increase_flow(source, target, capacity, flow, trace);
	}
	
	std::vector<int> result;
	result.clear();
	for (int n1 = 0; n1 < N1; ++n1) {
		int count = 0;
		for (int n2 = 0; n2 < N2; ++n2) {
			if (flow[n1][N1 + n2] > 0) {
				result.push_back(n2);
				++count;
			}
		}
		assert(count = 1);
	}
	
	for (int u = 0; u < N; ++u) {
		delete[] cost[u];
		delete[] capacity[u];
		delete[] flow[u];
	}
	delete[] cost;
	delete[] capacity;
	delete[] flow;

	delete[] queue;
	delete[] in_queue;
	delete[] d;
	delete[] trace;

	return result;
}

// +----------------+
// | Memory copying |
// +----------------+

static void transport(
	const int B, 
	const int N, 
	const int C, 
	const int b, 
	const std::vector<int> &result, 
	float *perm_flat) 
{
	assert(result.size() == N);
	for (int n1 = 0; n1 < N; ++n1) {
		const int n2 = result[n1];
		const int index = Index(b, n1, n2, N, N); 
		perm_flat[index] = 1.0;
	}
}

static void transport(
	const int B, 
	const int N, 
	const int C, 
	const int b, 
	const int c,
	const std::vector<int> &result, 
	float *perm_flat) 
{
	assert(result.size() == N);
	for (int n1 = 0; n1 < N; ++n1) {
		const int n2 = result[n1];
		const int index = Index(b, n1, n2, c, N, N, C); 
		perm_flat[index] = 1.0;
	}
}

// +---------------------------------------------------------+
// | Finding the permutation by Hungarian matching algorithm |
// +---------------------------------------------------------+

static void find_perm(
	const int B, 
	const int N, 
	const int C, 
	const int b, 
	const float *mu_encoder_flat, 
	const float *mu_prior_flat, 
	float *perm_flat,
	const bool separate_channels) 
{
	if (separate_channels == 1) {
		for (int c = 0; c < C; ++c) {
			const std::vector<std::vector<float>> cost = L1_cost_matrix(B, N, C, b, c, mu_encoder_flat, mu_prior_flat);
			const std::vector<int> result = Maximum_Flow_Minimum_Cost(cost);
			transport(B, N, C, b, c, result, perm_flat);
		}
	} else {
		const std::vector<std::vector<float>> cost = L1_cost_matrix(B, N, C, b, mu_encoder_flat, mu_prior_flat);
		const std::vector<int> result = Maximum_Flow_Minimum_Cost(cost);
		transport(B, N, C, b, result, perm_flat);
	}
}

void bipartite_matching(
	const at::Tensor &mu_encoder, 
	const at::Tensor &mu_prior, 
	at::Tensor &perm, 
	const int nThreads,
	const int separate_channels) 
{
	assert(mu_encoder.dim() == 3);
	assert(mu_prior.dim() == 3);

	// Batch size
	const int B = mu_encoder.size(0);

	// Number of vertices
	const int N = mu_encoder.size(1);

	// Number of channels
	const int C = mu_encoder.size(2);

	assert(mu_prior.size(0) == B);
	assert(mu_prior.size(1) == N);
	assert(mu_prior.size(2) == C);

	if (separate_channels == 1) {
		assert(perm.dim() == 4);
		assert(perm.size(0) == B);
		assert(perm.size(1) == N);
		assert(perm.size(2) == N);
		assert(perm.size(3) == C);
	} else {
		assert(perm.dim() == 3);
		assert(perm.size(0) == B);
		assert(perm.size(1) == N);
		assert(perm.size(2) == N);
	}

	float *mu_encoder_flat = reinterpret_cast<float*>(mu_encoder.data<float>());
	float *mu_prior_flat = reinterpret_cast<float*>(mu_prior.data<float>());
	float *perm_flat = reinterpret_cast<float*>(perm.data<float>());

	if (nThreads == 0) {
		for (int b = 0; b < B; ++b) {
			find_perm(B, N, C, b, mu_encoder_flat, mu_prior_flat, perm_flat, separate_channels);
		}
	} else {
		assert(nThreads >= 1);
		std::thread *job = new std::thread [nThreads];
		int start = 0;
		while (start < B) {
			const int finish = min(start + nThreads, B) - 1;
			for (int b = start; b <= finish; ++b) {
				job[b - start] = std::thread(
					find_perm, B, N, C, b, mu_encoder_flat, mu_prior_flat, perm_flat, separate_channels
				);
			}
			for (int b = start; b <= finish; ++b) {
				job[b - start].join();
			}
			start = finish + 1;
		}
		delete[] job;
	}
}

static void find_perm_2(
	const int B, 
	const int N, 
	const int C, 
	const int b, 
	const float *mu_encoder_flat, 
	const float *mu_prior_flat, 
	const float *logvar_encoder_flat, 
	const float *logvar_prior_flat, 
	float *perm_flat,
	const int separate_channels) 
{
	if (separate_channels == 1) {
		for (int c = 0; c < C; ++c) {
			const std::vector<std::vector<float>> cost = KL_cost_matrix(B, N, C, b, c, mu_encoder_flat, mu_prior_flat, logvar_encoder_flat, logvar_prior_flat);
			const std::vector<int> result = Maximum_Flow_Minimum_Cost(cost);
			transport(B, N, C, b, c, result, perm_flat);
		}
	} else {
		const std::vector<std::vector<float>> cost = KL_cost_matrix(B, N, C, b, mu_encoder_flat, mu_prior_flat, logvar_encoder_flat, logvar_prior_flat);
		const std::vector<int> result = Maximum_Flow_Minimum_Cost(cost);
		transport(B, N, C, b, result, perm_flat);
	}
}

void bipartite_matching_2(
	const at::Tensor &mu_encoder, 
	const at::Tensor &mu_prior, 
	const at::Tensor &logvar_encoder, 
	const at::Tensor &logvar_prior, 
	at::Tensor &perm, 
	const int nThreads,
	const int separate_channels) 
{
	assert(mu_encoder.dim() == 3);
	assert(mu_prior.dim() == 3);
	assert(logvar_encoder.dim() == 3);
	assert(logvar_prior.dim() == 3);

	// Batch size
	const int B = mu_encoder.size(0);

	// Number of vertices
	const int N = mu_encoder.size(1);

	// Number of channels
	const int C = mu_encoder.size(2);

	assert(logvar_encoder.size(0) == B);
	assert(logvar_encoder.size(1) == N);
	assert(logvar_encoder.size(2) == C);

	assert(mu_prior.size(0) == B);
	assert(mu_prior.size(1) == N);
	assert(mu_prior.size(2) == C);

	assert(logvar_prior.size(0) == B);
	assert(logvar_prior.size(1) == N);
	assert(logvar_prior.size(2) == C);

	if (separate_channels == 1) {
		assert(perm.dim() == 4);
		assert(perm.size(0) == B);
		assert(perm.size(1) == N);
		assert(perm.size(2) == N);
		assert(perm.size(3) == C);
	} else {
		assert(perm.dim() == 3);
		assert(perm.size(0) == B);
		assert(perm.size(1) == N);
		assert(perm.size(2) == N);
	}

	float *mu_encoder_flat = reinterpret_cast<float*>(mu_encoder.data<float>());
	float *mu_prior_flat = reinterpret_cast<float*>(mu_prior.data<float>());
	float *logvar_encoder_flat = reinterpret_cast<float*>(logvar_encoder.data<float>());
	float *logvar_prior_flat = reinterpret_cast<float*>(logvar_prior.data<float>());
	float *perm_flat = reinterpret_cast<float*>(perm.data<float>());

	if (nThreads == 0) {
		for (int b = 0; b < B; ++b) {
			find_perm_2(B, N, C, b, mu_encoder_flat, mu_prior_flat, logvar_encoder_flat, logvar_prior_flat, perm_flat, separate_channels);
		}
	} else {
		assert(nThreads >= 1);
		std::thread *job = new std::thread [nThreads];
		int start = 0;
		while (start < B) {
			const int finish = min(start + nThreads, B) - 1;
			for (int b = start; b <= finish; ++b) {
				job[b - start] = std::thread(
					find_perm_2, B, N, C, b, mu_encoder_flat, mu_prior_flat, logvar_encoder_flat, logvar_prior_flat, perm_flat, separate_channels
				);
			}
			for (int b = start; b <= finish; ++b) {
				job[b - start].join();
			}
			start = finish + 1;
		}
		delete[] job;
	}
}

static void find_perm_3(
	const int B, 
	const int N, 
	const int b, 
	const float *cost_matrix_flat, 
	float *perm_flat
) 
{
	
	std::vector<std::vector<float>> cost;
	cost.clear();
	for (int i = 0; i < N; ++i) {
		std::vector<float> row;
		row.clear();
		for (int j = 0; j < N; ++j) {
			const int index = Index(b, i, j, N, N);
			row.push_back(cost_matrix_flat[index]);
		}
		cost.push_back(row);
	}
	const std::vector<int> result = Maximum_Flow_Minimum_Cost(cost);
	assert(result.size() == N);
	for (int n1 = 0; n1 < N; ++n1) {
		const int n2 = result[n1];
		const int index = Index(b, n1, n2, N, N); 
		perm_flat[index] = 1.0;
	}
}

void bipartite_matching_3(
	const at::Tensor &cost_matrix, 
	at::Tensor &perm, 
	const int nThreads
)
{
	assert(cost_matrix.dim() == 3);
	assert(perm.dim() == 3);

	// Batch size
	const int B = cost_matrix.size(0);

	// Number of vertices
	const int N = cost_matrix.size(1);

	assert(cost_matrix.size(2) == N);
	assert(perm.size(0) == B);
	assert(perm.size(1) == N);
	assert(perm.size(2) == N);

	float *cost_matrix_flat = reinterpret_cast<float*>(cost_matrix.data<float>());
	float *perm_flat = reinterpret_cast<float*>(perm.data<float>());

	if (nThreads == 0) {
		for (int b = 0; b < B; ++b) {
			find_perm_3(B, N, b, cost_matrix_flat, perm_flat);
		}
	} else {
		assert(nThreads >= 1);
		std::thread *job = new std::thread [nThreads];
		int start = 0;
		while (start < B) {
			const int finish = min(start + nThreads, B) - 1;
			for (int b = start; b <= finish; ++b) {
				job[b - start] = std::thread(
					find_perm_3, B, N, b, cost_matrix_flat, perm_flat
				);
			}
			for (int b = start; b <= finish; ++b) {
				job[b - start].join();
			}
			start = finish + 1;
		}
		delete[] job;
	}
}

// +-------------------------------------------------------------+
// | Free matching using L1 distance for each channel separately |
// +-------------------------------------------------------------+

static std::vector<int> select_min_row_wise(const std::vector<std::vector<float>> cost) {
	std::vector<int> result;
	result.clear();
	for (int i = 0; i < cost.size(); ++i) {
		int best = 0;
		for (int j = 1; j < cost[i].size(); ++j) {
			if (cost[i][j] < cost[i][best]) {
				best = j;
			}
		}
		result.push_back(best);
	}
	return result;
}

static void find_free_perm(
	const int B, 
	const int N, 
	const int C, 
	const int b, 
	const float *mu_encoder_flat, 
	const float *mu_prior_flat, 
	float *perm_flat,
	const int separate_channels) 
{
	if (separate_channels == 1) {
		for (int c = 0; c < C; ++c) {
			const std::vector<std::vector<float>> cost = L1_cost_matrix(B, N, C, b, c, mu_encoder_flat, mu_prior_flat);
			const std::vector<int> result = select_min_row_wise(cost);
			transport(B, N, C, b, c, result, perm_flat);
		}
	} else {
		const std::vector<std::vector<float>> cost = L1_cost_matrix(B, N, C, b, mu_encoder_flat, mu_prior_flat);
		const std::vector<int> result = select_min_row_wise(cost);
		transport(B, N, C, b, result, perm_flat);
	}
}

void free_matching(
	const at::Tensor &mu_encoder, 
	const at::Tensor &mu_prior, 
	at::Tensor &perm,
	const int nThreads,
	const int separate_channels) 
{
	assert(mu_encoder.dim() == 3);
	assert(mu_prior.dim() == 3);

	// Batch size
	const int B = mu_encoder.size(0);

	// Number of vertices
	const int N = mu_encoder.size(1);

	// Number of channels
	const int C = mu_encoder.size(2);

	assert(mu_prior.size(0) == B);
	assert(mu_prior.size(1) == N);
	assert(mu_prior.size(2) == C);

	if (separate_channels == 1) {
		assert(perm.dim() == 4);
		assert(perm.size(0) == B);
		assert(perm.size(1) == N);
		assert(perm.size(2) == N);
		assert(perm.size(3) == C);
	} else {
		assert(perm.dim() == 3);
		assert(perm.size(0) == B);
		assert(perm.size(1) == N);
		assert(perm.size(2) == N);
	}

	float *mu_encoder_flat = reinterpret_cast<float*>(mu_encoder.data<float>());
	float *mu_prior_flat = reinterpret_cast<float*>(mu_prior.data<float>());
	float *perm_flat = reinterpret_cast<float*>(perm.data<float>());
	
	if (nThreads == 0) {
		for (int b = 0; b < B; ++b) {
			find_free_perm(B, N, C, b, mu_encoder_flat, mu_prior_flat, perm_flat, separate_channels);
		}
	} else {
		std::thread *job = new std::thread [nThreads];
		int start = 0;
		while (start < B) {
			const int finish = min(start + nThreads, B) - 1;
			for (int b = start; b <= finish; ++b) {
				job[b - start] = std::thread(
					find_free_perm, B, N, C, b, mu_encoder_flat, mu_prior_flat, perm_flat, separate_channels
				);
			}
			for (int b = start; b <= finish; ++b) {
				job[b - start].join();
			}
			start = finish + 1;
		}
		delete[] job;
	}
}

// +------------------------------------------------------------------------+
// | Free matching using KL-divergence distance for each channel separately |
// +------------------------------------------------------------------------+

static void find_free_perm_2(
	const int B, 
	const int N, 
	const int C, 
	const int b, 
	const float *mu_encoder_flat, 
	const float *mu_prior_flat, 
	const float *logvar_encoder_flat,
	const float *logvar_prior_flat,
	float *perm_flat,
	const int separate_channels) 
{
	if (separate_channels == 1) {
		for (int c = 0; c < C; ++c) {
			const std::vector<std::vector<float>> cost = KL_cost_matrix(B, N, C, b, c, mu_encoder_flat, mu_prior_flat, logvar_encoder_flat, logvar_prior_flat);
			const std::vector<int> result = select_min_row_wise(cost);
			transport(B, N, C, b, c, result, perm_flat);
		}
	} else {
		const std::vector<std::vector<float>> cost = KL_cost_matrix(B, N, C, b, mu_encoder_flat, mu_prior_flat, logvar_encoder_flat, logvar_prior_flat);
		const std::vector<int> result = select_min_row_wise(cost);
		transport(B, N, C, b, result, perm_flat);
	}
}

void free_matching_2(
	const at::Tensor &mu_encoder, 
	const at::Tensor &mu_prior, 
	const at::Tensor &logvar_encoder,
	const at::Tensor &logvar_prior,
	at::Tensor &perm,
	const int nThreads,
	const int separate_channels) 
{
	assert(mu_encoder.dim() == 3);
	assert(mu_prior.dim() == 3);
	assert(logvar_encoder.dim() == 3);
	assert(logvar_prior.dim() == 3);

	// Batch size
	const int B = mu_encoder.size(0);

	// Number of vertices
	const int N = mu_encoder.size(1);

	// Number of channels
	const int C = mu_encoder.size(2);

	assert(mu_prior.size(0) == B);
	assert(mu_prior.size(1) == N);
	assert(mu_prior.size(2) == C);

	assert(logvar_encoder.size(0) == B);
	assert(logvar_encoder.size(1) == N);
	assert(logvar_encoder.size(2) == C);

	assert(logvar_prior.size(0) == B);
	assert(logvar_prior.size(1) == N);
	assert(logvar_prior.size(2) == C);

	if (separate_channels == 1) {
		assert(perm.dim() == 4);
		assert(perm.size(0) == B);
		assert(perm.size(1) == N);
		assert(perm.size(2) == N);
		assert(perm.size(3) == C);
	} else {
		assert(perm.dim() == 3);
		assert(perm.size(0) == B);
		assert(perm.size(1) == N);
		assert(perm.size(2) == N);
	}

	float *mu_encoder_flat = reinterpret_cast<float*>(mu_encoder.data<float>());
	float *mu_prior_flat = reinterpret_cast<float*>(mu_prior.data<float>());
	float *logvar_encoder_flat = reinterpret_cast<float*>(logvar_encoder.data<float>());
	float *logvar_prior_flat = reinterpret_cast<float*>(logvar_prior.data<float>());
	float *perm_flat = reinterpret_cast<float*>(perm.data<float>());
	
	if (nThreads == 0) {
		for (int b = 0; b < B; ++b) {
			find_free_perm_2(B, N, C, b, mu_encoder_flat, mu_prior_flat, logvar_encoder_flat, logvar_prior_flat, perm_flat, separate_channels);
		}
	} else {
		std::thread *job = new std::thread [nThreads];
		int start = 0;
		while (start < B) {
			const int finish = min(start + nThreads, B) - 1;
			for (int b = start; b <= finish; ++b) {
				job[b - start] = std::thread(
					find_free_perm_2, B, N, C, b, mu_encoder_flat, mu_prior_flat, logvar_encoder_flat, logvar_prior_flat, perm_flat, separate_channels
				);
			}
			for (int b = start; b <= finish; ++b) {
				job[b - start].join();
			}
			start = finish + 1;
		}
		delete[] job;
	}
}

std::vector<at::Tensor> test_api(const std::vector<at::Tensor> &tensors) {
	const int N = tensors.size();
	std::vector<at::Tensor> result;
	for (int i = 0; i < N; ++i) {
		result.push_back(torch::zeros({}));
	}
	return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("bipartite_matching", &bipartite_matching, "Bipartite matching for mu only (L1 distance)");
	m.def("bipartite_matching_2", &bipartite_matching_2, "Bipartite matching for both mu and logvar (KL-divergence distance)");
	m.def("free_matching", &free_matching, "Free matching (L1 distance)");
	m.def("free_matching_2", &free_matching_2, "Free matching (KL-divergence distance)");
	m.def("test_api", &test_api, "Test API");
	m.def("bipartite_matching_3", &bipartite_matching_3, "Bipartite matching given cost matrix");
}
