import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
from scipy import io
from torch.utils.data import DataLoader


class Cub(Dataset):
	"""
	images.mat['images'][0,0]: 1-11788
	images.mat['images'][0,1]:  [[array(['001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg'],
      dtype='<U61')],
       [array(['001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg'],
      dtype='<U61')], ... ]
	image_class_labels['imageClassLabels']: array([[    1,     1],
											       [    2,     1],
											       [    3,     1],
											       ...,
											       [11786,   200],
											       [11787,   200],
											       [11788,   200]], dtype=int32), shape = [11788, 2]
	class_attribute_labels_continuous['classAttributes'].shape= (200, 312)

	"""
	def __init__(self, root, n_way, k_query, train = True, episode_num = 1000):
		"""
		Actually, the image here act as query role. we want to find the closed attribute item for each image.
		:param root:
		:param n_way:
		:param k_query:
		:param train:
		:param episode_num:
		"""
		super(Cub, self).__init__()

		self.root = root
		self.n_way = n_way
		self.k_query = k_query
		self.episode_num = episode_num
		self.train = train

		# load mat file.
		self.image_class_labels = io.loadmat(os.path.join(root, 'image_class_labels.mat'))
		self.image_class_labels = self.image_class_labels['imageClassLabels'][:, 1]
		self.image_class_labels = self.image_class_labels.reshape(11788)
		# self.image_class_labels[8822] = 151
		# print('>>image_class_labels:', self.image_class_labels[8820:8830])

		self.image_features = io.loadmat(os.path.join(root, 'cnn_feat-imagenet-vgg-verydeep-19.mat'))
		self.image_features = self.image_features['cnn_feat'].swapaxes(0, 1)
		# from index 8822, its label start from 151
		self.image_features = self.image_features[:8822] if train else self.image_features[8822:]
		# [11788, 4096]
		print('img features:', self.image_features.shape)


		self.images = io.loadmat(os.path.join(root, 'images.mat'))
		self.images = self.images['images'][0, 1]
		self.images = np.array(self.images.tolist()).squeeze(2).squeeze(1).reshape(11788)
		# flatten [path1, path2, ....]
		# print('>>images:', self.images)

		images_by_cls = []
		for i in range(200):
			num = self.images[np.equal(self.image_class_labels, i + 1)]
			images_by_cls.append(num)
		# gather db by same label: [[label1_img1, label1_img2,...], [label2_img1, label2_img2,...], ...]
		# each class has different num of imgs, here we use a list to save it.
		self.images_by_cls = images_by_cls[:150] if train else images_by_cls[150:]
		# print('>>img group by cls:', len(self.images_by_cls))

		self.class_attributes = io.loadmat(os.path.join(root, 'class_attribute_labels_continuous.mat'))
		self.class_attributes = self.class_attributes['classAttributes'].reshape(200, 312).astype(np.float32)
		self.class_attributes = self.class_attributes[:150] if train else self.class_attributes[150:]
		# print('>>class_attributes:', self.class_attributes.shape)

		self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
		                                     transforms.Resize((299, 299)),
		                                     transforms.ToTensor(),
		                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		                                     ])


	def __getitem__(self, item):

		# randomly sample n-way classes from train/test set
		# [5, 29, 39, 1, 129...]
		selected_cls_label = np.random.choice(range(len(self.images_by_cls)), self.n_way, False)
		# select all imgs for the selected categories, its a list.
		# [[5_img1, 5_img2,...], [29_img1, 29_img2,...],...]
		selected_img_by_cls = [self.images_by_cls[i] for i in selected_cls_label]
		# only sample one for each category, [[5_img1, 5_img2], [29_img1, 29_img2], ....]
		# [n_way, k_query]
		selected_imgs = [np.random.choice(imgs, self.k_query, False) for imgs in selected_img_by_cls]

		# select attributes for each class
		# [n_way, 312]
		selected_atts = self.class_attributes[selected_cls_label]


		# [n_way, k_query] => [setsz=n_way*k_query]
		selected_imgs = np.array(selected_imgs).reshape(-1)

		x = []
		for img in selected_imgs:
			# find the idx in images list and get corresponding features from features list.
			idx = np.where(self.images == img)[0][0]
			if not self.train: idx -= 8822

			feature = self.image_features[idx]
			x.append(torch.from_numpy(feature))
		x = torch.stack(x)
		att = torch.from_numpy(selected_atts)
		att_label = torch.from_numpy(selected_cls_label)
		# [n_way] => [n_way, 1] => [n_way, k_query] => [n_way*k_query]
		x_label = att_label.clone().unsqueeze(1).repeat(1, self.k_query).view(-1)


		# shuffle
		shuffle_idx = torch.randperm(self.n_way * self.k_query)
		x = x[shuffle_idx]
		x_label = x_label[shuffle_idx]


		# print('selected_imgs', np.array(selected_imgs)[shuffle_idx][:5])
		# print('imgs:', x.size())
		# print('attrs:', att.size(), att[:5])
		# print('att label:', att_label.numpy())
		# print('x label:', x_label.numpy())


		return x, x_label, att, att_label



	def __len__(self):
		return self.episode_num





def test():
	db = Cub('../CUB_200_2011_ZL/', 50, 2, train=False)

	db_loader = DataLoader(db, 2, True, num_workers=2, pin_memory=True)

	iter(db_loader).next()


if __name__ == '__main__':
	test()