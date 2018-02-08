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
	def __init__(self, root, n_way, k_query, train = True, episode_num = 10000):
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

		if train:

			self.x = io.loadmat(os.path.join(root, 'train_cub_googlenet_bn.mat'))
			# (8855, 1024)
			self.x = self.x['train_cub_googlenet_bn']

			# (8855, 312)
			self.att = io.loadmat(os.path.join(root, 'train_attr.mat'))
			self.att = self.att['train_attr']


			x_label = [0]
			att_cls_label = [0]
			att_cls = [self.att[0]]
			for idx in range(self.att.shape[0]):
				if idx + 1 == self.att.shape[0]:
					break
				if np.array_equal(self.att[idx+1], self.att[idx]):
					x_label.append(x_label[-1])
				else:
					x_label.append(x_label[-1] + 1)
					# x_label has been updated!
					att_cls_label.append(x_label[-1])
					att_cls.append(self.att[idx + 1])

			x_label = np.array(x_label)
			att_cls_label = np.array(att_cls_label)
			att_cls = np.array(att_cls)
			# print('x_label', x_label.shape)
			# print('att label:', att_cls_label.shape)
			# print('att:', att_cls.shape)

			self.x_label = x_label

			for idx, x in enumerate(self.x):
				x_label = self.x_label[idx]
				att_in_x = self.att[idx]

				idx_in_att = np.where(att_cls_label == x_label)[0][0]
				att_in_att = att_cls[idx_in_att]
				att_label_in_att = att_cls_label[idx_in_att]

				assert x_label == att_label_in_att
				assert np.array_equal(att_in_x, att_in_att)
			print('att in x match att in att_cls!!')

			self.att = att_cls
			self.att_label = att_cls_label

		else:
			self.x = io.loadmat(os.path.join(root, 'test_cub_googlenet_bn.mat'))
			# (2933, 1024)
			self.x = self.x['test_cub_googlenet_bn']

			# (50, 312)
			self.att = io.loadmat(os.path.join(root, 'test_proto.mat'))
			self.att = self.att['test_proto']

			# (2933,)
			self.x_label = io.loadmat(os.path.join(root, 'test_labels_cub.mat'))
			self.x_label = self.x_label['test_labels_cub'].reshape(-1)

			# (50,)
			self.att_label = io.loadmat(os.path.join(root, 'testclasses_id.mat'))
			self.att_label = self.att_label['testclasses_id'].reshape(-1)


	def __getitem__(self, item):

		# randomly sample n-way classes from train/test set
		# [n-way]
		selected_cls_idx = np.random.choice(range(self.att_label.shape[0]), self.n_way, False)

		# selected cls label and att
		# [n_way, 312]
		selected_att = self.att[selected_cls_idx]
		selected_att_label = self.att_label[selected_cls_idx]

		selected_x = []
		selected_x_label = []
		for att_label in selected_att_label:
			# randomly select 1 img which has the same label.
			idx = np.random.choice(np.where(self.x_label == att_label)[0], 1)[0]
			# get the img's features and img's label
			f = self.x[idx]
			f_label = self.x_label[idx]

			selected_x.append(f)
			selected_x_label.append(f_label)

		# selected_x_label has the order with selected_att_label here.
		selected_x = np.array(selected_x)
		selected_x_label = np.array(selected_x_label)


		x = torch.from_numpy(selected_x).float()
		x_label = torch.from_numpy(selected_x_label).long()
		att = torch.from_numpy(selected_att).float()
		att_label = torch.from_numpy(selected_att_label).long()

		# shuffle x & x_label in case it owns the same order with att_label
		shuffle_idx = torch.randperm(self.n_way)
		# x = x[shuffle_idx]
		# x_label = x_label[shuffle_idx]

		# print('selected_imgs', np.array(selected_imgs)[shuffle_idx][:5])
		# print('imgs:', x.size())
		# print('attrs:', att.size(), att[:5])
		# print('att label:', att_label.numpy())
		# print('x label:', x_label.numpy())


		return x, x_label, att, att_label



	def __len__(self):
		return self.episode_num





def test():
	db = Cub('../CUB_data/', 50, 2, train=True)

	db_loader = DataLoader(db, 2, True, num_workers=2, pin_memory=True)

	iter(db_loader).next()


if __name__ == '__main__':
	test()
