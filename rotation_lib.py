from PIL import Image, ImageDraw
import math
import random
import numpy as np

random_seed = 123456789
random.seed(random_seed)

class Rotation_Image_Dataset:
	def __init__(self, image_fn, num_rotations, num_landmarks, window_size):
		self.num_rotations = num_rotations
		self.num_landmarks = num_landmarks
		self.window_size = window_size
		self.angles_degree = [i * 360 / self.num_rotations for i in range(self.num_rotations)]
		self.theta = 2 * math.pi / num_rotations
		self.angles = [i * self.theta for i in range(self.num_rotations)]

		self.image = Image.open(image_fn)
		self.width = self.image.width
		self.height = self.image.height
		self.rotated_images = []
		for angle in self.angles_degree:
			self.rotated_images.append(self.image.rotate(angle))

		self.x0 = self.width / 2
		self.y0 = self.height / 2
		self.landmark = np.zeros((self.num_landmarks, 2))
		for i in range(self.num_landmarks):
			self.landmark[i, 0] = random.randint(0, self.width) - self.x0
			self.landmark[i, 1] = random.randint(0, self.height) - self.y0
		
		self.rotation_matrix = np.zeros((2, 2))
		self.rotation_matrix[0, 0] = math.cos(self.theta)
		self.rotation_matrix[0, 1] = math.sin(self.theta)
		self.rotation_matrix[1, 0] = - math.sin(self.theta)
		self.rotation_matrix[1, 1] = math.cos(self.theta)

		self.landmarks = []
		for i in range(self.num_rotations):
			if i == 0:
				self.landmarks.append(self.landmark)
			else:
				new = np.matmul(self.landmarks[i - 1], self.rotation_matrix)
				self.landmarks.append(new)

		self.features = []
		for i in range(self.num_rotations):
			for j in range(self.num_landmarks):
				self.landmarks[i][j, 0] += self.x0
				self.landmarks[i][j, 1] += self.y0 
				self.landmarks[i][j, 1] = self.height - self.landmarks[i][j, 1] - 1
			feature = self.get_feature(self.rotated_images[i], self.landmarks[i], self.window_size)
			self.features.append(feature)

	def get_feature(self, image, landmark, window_size):
		rgb = image.convert("RGB")
		num_landmarks = landmark.shape[0]
		feature = np.zeros((num_landmarks, 3))
		for i in range(num_landmarks):
			x = int(landmark[i, 0])
			y = int(landmark[i, 1])
			count = 0
			for u in range(x - window_size, x + window_size):
				for v in range(y - window_size, y + window_size):
					if u >= 0 and u < image.width and v >= 0 and v < image.height:
						r, g, b = rgb.getpixel((u, v))
						count += 1
			if count > 0:
				feature[i, 0] = r / count
				feature[i, 1] = g / count
				feature[i, 2] = b / count
		return feature

	def draw(self, images = None, landmarks = None, ground_truth = None, match = None):
		if images is None:
			images = self.rotated_images + [self.rotated_images[0]]
		
		if landmarks is None:
			landmarks = self.landmarks + [self.landmarks[0]]
		
		width = 0
		for image in images:
			width += image.width
		
		height = images[0].height
		result = Image.new('RGB', (width, height))
		
		shift = 0
		for i in range(len(images)):
			result.paste(images[i], (shift, 0))
			shift += images[i].width

		draw = ImageDraw.Draw(result)
		shift = 0
		for i in range(len(images)):
			num_landmarks = landmarks[i].shape[0]
			for j in range(num_landmarks):
				x = int(landmarks[i][j, 0] + shift)
				y = int(landmarks[i][j, 1])

				r = 5
				w = 3
				draw.line((x - r, y - r, x + r, y + r), fill = (0, 255, 0), width = w)
				draw.line((x + r, y - r, x - r, y + r), fill = (0, 255, 0), width = w)

				"""
				r = 20
				w = 1
				draw.line((x - r, y - r, x + r, y - r), fill = (0, 255, 0), width = w)
				draw.line((x + r, y - r, x + r, y + r), fill = (0, 255, 0), width = w)
				draw.line((x + r, y + r, x - r, y + r), fill = (0, 255, 0), width = w)
				draw.line((x - r, y + r, x - r, y - r), fill = (0, 255, 0), width = w)
				"""

			# Draw matching
			if i > 0:
				if ground_truth is not None:
					for j in range(num_landmarks):
						xx = int(landmarks[i - 1][j, 0] + shift - images[i - 1].width)
						yy = int(landmarks[i - 1][j, 1])
						
						for v in range(num_landmarks):
							if ground_truth[i - 1][j, v] == 1:
								x = int(landmarks[i][v, 0] + shift)
								y = int(landmarks[i][v, 1])

								if x >= 0 and x < shift + images[i].width:
									if y >= 0 and y < images[i].height:
										if xx >= 0 and xx < shift:
											if yy >= 0 and yy < images[i].height:
												draw.line((x, y, xx, yy), fill = (0, 255, 0), width = w)

				if match is not None:
					for j in range(num_landmarks):
						xx = int(landmarks[i - 1][j, 0] + shift - images[i - 1].width)
						yy = int(landmarks[i - 1][j, 1])
						
						for v in range(num_landmarks):
							if match[i - 1][j, v] == 1:
								x = int(landmarks[i][v, 0] + shift)
								y = int(landmarks[i][v, 1])

								if x >= 0 and x < shift + images[i].width:
									if y >= 0 and y < images[i].height:
										if xx >= 0 and xx < shift:
											if yy >= 0 and yy < images[i].height:
												draw.line((x, y, xx, yy), fill = (255, 0, 0), width = w)

			shift += images[i].width

		return result