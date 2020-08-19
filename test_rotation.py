from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

import hungarian_lib
from rotation_lib import Rotation_Image_Dataset

def error(T, num_rotations, num_landmarks):
	error = 0
	for i in range(num_rotations):
		for j in range(num_rotations):
			if i < j:
				error += np.sum(abs(T[i, j] - np.eye(num_landmarks))) / 2
	return error

num_rotations = 8
num_landmarks = 5
window_size = 5
num_features = 3 # Red, Green, Blue
obj = Rotation_Image_Dataset("jim_simons.jpg", num_rotations, num_landmarks, window_size)

T = np.zeros((num_rotations, num_rotations, num_landmarks, num_landmarks))

for i in range(num_rotations):
	for j in range(num_rotations):
		if i == j:
			T[i, j] = np.eye(num_landmarks)
		else:
			if i < j:
				feature_x = torch.from_numpy(obj.features[i]).reshape(1, num_landmarks, num_features).type(torch.FloatTensor)
				feature_y = torch.from_numpy(obj.features[j]).reshape(1, num_landmarks, num_features).type(torch.FloatTensor)
				perm = torch.zeros(1, num_landmarks, num_landmarks)
				hungarian_lib.bipartite_matching(feature_x, feature_y, perm, 20, 0)
				T[i, j] = perm.reshape(num_landmarks, num_landmarks).numpy()
				T[j, i] = np.transpose(T[i, j])

# +-----------------------------+
# | Permutation synchronization |
# +-----------------------------+

T_ = np.zeros((num_rotations * num_landmarks, num_rotations * num_landmarks))
for i in range(num_rotations):
	for j in range(num_rotations):
		T_[i * num_landmarks : (i + 1) * num_landmarks, j * num_landmarks : (j + 1) * num_landmarks] = T[i, j]

values, vectors = np.linalg.eig(T_ + 1e-4 * np.eye(num_rotations * num_landmarks))

values = - np.abs(values)
sorted_index = np.argsort(values)

U = np.sqrt(num_rotations) * vectors[:, sorted_index[:num_landmarks]]
sigma = []
for i in range(num_rotations):
	A = U[i * num_landmarks : (i + 1) * num_landmarks, :]
	B = U[: num_landmarks, : num_landmarks].transpose()
	P = np.matmul(A, B).real
	cost_matrix = torch.from_numpy(P).reshape(1, num_landmarks, num_landmarks).type(torch.FloatTensor)
	cost_matrix = torch.max(cost_matrix) - cost_matrix
	perm = torch.zeros(1, num_landmarks, num_landmarks)
	hungarian_lib.bipartite_matching_3(cost_matrix, perm, 1)
	sigma.append(perm.reshape(num_landmarks, num_landmarks).numpy())

tau = np.zeros((num_rotations, num_rotations, num_landmarks, num_landmarks))
for i in range(num_rotations):
	for j in range(num_rotations):
		tau[i, j] = np.matmul(sigma[i], np.linalg.inv(sigma[j]))

synchronized_error = error(tau, num_rotations, num_landmarks)
print("Permutation synchronization: ", synchronized_error)

# +----------+
# | Baseline |
# +----------+

baseline_error = error(T, num_rotations, num_landmarks)
print("Baseline error = ", baseline_error)

improved_percent = (baseline_error - synchronized_error) / baseline_error * 100;
print("Improved percentage of matching by perm synch = ", improved_percent)

def draw_series(T, file_name):
	# Drawing
	images = []
	landmarks = []
	ground_truth = []
	match = []
	for i in range(num_rotations):
		images.append(obj.rotated_images[i])
		landmarks.append(obj.landmarks[i])
		if i + 1 < num_rotations:
			ground_truth.append(np.eye(num_landmarks))
			match.append(T[i, i + 1])
		else:
			ground_truth.append(np.eye(num_landmarks))
			match.append(T[i, 0])
	images.append(obj.rotated_images[0])
	landmarks.append(obj.landmarks[0])

	picture = obj.draw(images, landmarks, ground_truth, match)
	picture.save(file_name, "PNG")
	picture.show()

draw_series(T, "baseline_series.png")
draw_series(tau, "synch_series.png")

# Search for the most contrastive pair
best_constrast = 0
for i in range(num_rotations):
	for j in range(num_rotations):
		synch_error = np.sum(np.abs(tau[i, j] - np.eye(num_landmarks)))
		baseline_error = np.sum(np.abs(T[i, j] - np.eye(num_landmarks)))
		if baseline_error - synch_error >= best_constrast:
			best_constrast = baseline_error - synch_error
			x = i
			y = j

# Draw the best constract pair
picture = obj.draw([obj.rotated_images[x], obj.rotated_images[y]], [obj.landmarks[x], obj.landmarks[y]], [np.eye(num_landmarks)], [T[x, y]])
picture.save("baseline.png", "PNG")

# picture.show()
plt.subplot(1, 2, 1)
plt.imshow(picture)
plt.title("Baseline matching (Hungarian)")

picture = obj.draw([obj.rotated_images[x], obj.rotated_images[y]], [obj.landmarks[x], obj.landmarks[y]], [np.eye(num_landmarks)], [tau[x, y]])
picture.save("synch.png", "PNG")

# picture.show()
plt.subplot(1, 2, 2)
plt.imshow(picture)
plt.title("Permutation synchronized matching (spectral method)")
plt.show()

# print("Best constrast pair:", best_constrast)