import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from repnet import repnet_deep, Bottleneck


class RN(nn.Module):
	"""

	"""
	def __init__(self, n_way, k_shot):
		super(RN, self).__init__()

		self.n_way = n_way
		self.k_shot = k_shot

		self.repnet = nn.Sequential(repnet_deep(False), # (1024, 14, 14)
		                            nn.Conv2d(1024, 256, kernel_size=5, stride=3),
		                            nn.BatchNorm2d(256),
		                            nn.ReLU(inplace=True))
		# we need to know the feature dim, so here is a forwarding.
		repnet_sz = self.repnet(Variable(torch.rand(2, 3, 224, 224))).size()
		self.c = repnet_sz[1]
		self.d = repnet_sz[2]
		# this is the input channels of layer4&layer5
		self.inplanes = 2 * self.c
		assert repnet_sz[2] == repnet_sz[3]
		print('repnet sz:', repnet_sz)

		# the input is self.c with two coordination information, and then combine each
		self.g = nn.Sequential(nn.Linear( (self.c + 2) * 2, 256),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 256),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 256),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 256),
		                       nn.ReLU(inplace=True))

		self.f = nn.Sequential(nn.Linear(256, 256),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 256),
		                       nn.Dropout(),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 29),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(29, 1),
		                       nn.Sigmoid())

		coord = np.array([(i / self.d , j / self.d) for i in range(self.d) for j in range(self.d)])
		self.coord = torch.from_numpy(coord).float().view(self.d, self.d, 2).transpose(0, 2).transpose(1,2).contiguous()
		self.coord = self.coord.unsqueeze(0).unsqueeze(0)
		print('self.coord:', self.coord.size(),self.coord) # [batchsz:1, setsz:1, 2, self.d, self.d]



	def forward(self, support_x, support_y, query_x, query_y, train=True):
		"""

		:param support_x: [b, setsz, c_, h, w]
		:param support_y: [b, setsz]
		:param query_x:   [b, querysz, c_, h, w]
		:param query_y:   [b, querysz]
		:return:
		"""
		batchsz, setsz, c_, h, w = support_x.size()
		querysz = query_x.size(1)
		c, d = self.c, self.d

		# [b, setsz, c_, h, w] => [b*setsz, c_, h, w] => [b*setsz, c, d, d] => [b, setsz, c, d, d]
		support_xf = self.repnet(support_x.view(batchsz * setsz, c_, h, w)).view(batchsz, setsz, c, d, d)
		# [b, querysz, c_, h, w] => [b*querysz, c_, h, w] => [b*querysz, c, d, d] => [b, querysz, c, d, d]
		query_xf = self.repnet(query_x.view(batchsz * querysz, c_, h, w)).view(batchsz, querysz, c, d, d)

		## now make the combination between two pairs
		# include the coordinate information in each feature dim
		# [b, setsz/querysz, c+2, d, d]
		support_xf = torch.cat([support_xf, Variable(self.coord.expand(batchsz, setsz, 2, d, d)).cuda()], dim = 2)
		query_xf = torch.cat([query_xf, Variable(self.coord.expand(batchsz, querysz, 2, d, d)).cuda()], dim = 2)
		c += 2 # c is a copy of self.c, we need not reset c since it will be reseted in the beginning of forward

		# make combination now
		# [b, setsz, c, d, d] => [b, setsz, c, d*d] => [b, 1, setsz, c, d*d] => [b, 1, setsz, c, 1, d*d] => [b, querysz, setsz, c, d*d, d*d]
		support_xf = support_xf.view(batchsz, setsz, c, d*d).unsqueeze(1).unsqueeze(4).expand(batchsz, querysz, setsz, c, d*d, d*d)
		# [b, querysz, c, d, d] => [b, querysz, c, d*d] => [b, querysz, 1, c, d*d, 1] => [b, querysz, setsz, c, d*d, d*d]
		query_xf = query_xf.view(batchsz, querysz, c, d*d).unsqueeze(2).unsqueeze(5).expand(batchsz, querysz, setsz, c, d*d, d*d)
		# [b, querysz, setsz, c*2, d*d, d*d]
		comb = torch.cat([support_xf, query_xf], dim=3)

		# [b, querysz, setsz, c*2, d*d:0, d*d:1] => [b, querysz, setsz, d*d:1, d*d:0, c*2] => [b, querysz, setsz, d*d:0, d*d:1, c*2]
		comb = comb.transpose(3, 5).transpose(3, 4).contiguous().view(batchsz * querysz * setsz * d*d * d*d, c * 2)
		# push to G network
		# [b*querysz*setsz*d^4, 2c] => [b*querysz*setsz*d^4, 64]
		x_f = self.g(comb)
		# sum over coordinate axis and squeeze it
		# [b*querysz*setsz*d^4, 64] => [b*querysz*setsz*d^4, 64] => [b*querysz*setsz, d^4, 64] => [b*querysz*setsz, 64]
		x_f = x_f.view(batchsz * querysz * setsz, d*d * d*d, -1).sum(1) # the last dim can be derived by layer setting
		# push to F network
		# [batchsz * querysz * setsz, 64] => [batchsz * querysz * setsz, 1] => [batch, querysz, setsz]
		score = self.f(x_f).view(batchsz, querysz, setsz, 1).squeeze(3)

		# build its label
		# [b, setsz] => [b, 1, setsz] => [b, querysz, setsz]
		support_yf = support_y.unsqueeze(1).expand(batchsz, querysz, setsz)
		# [b, querysz] => [b, querysz, 1] => [b, querysz, setsz]
		query_yf = query_y.unsqueeze(2).expand(batchsz, querysz, setsz)
		# eq: [b, querysz, setsz] => [b, querysz, setsz] and convert byte tensor to float tensor
		label = torch.eq(support_yf, query_yf).float()

		# score: [b, querysz, setsz]
		# label: [b, querysz, setsz]
		if train:
			loss = torch.pow(label - score, 2).sum() / batchsz
			return loss

		else:
			# [b, querysz, setsz]
			rn_score_np = score.cpu().data.numpy()
			pred = []
			# [b, setsz]
			support_y_np = support_y.cpu().data.numpy()
			for i, batch in enumerate(rn_score_np):
				for j, query in enumerate(batch):
					# query: [setsz]
					sim = []  # [n_way]
					for way in range(self.n_way):
						sim.append(np.sum(query[way * self.k_shot: (way + 1) * self.k_shot]))
					idx = np.array(sim).argmax()
					pred.append(support_y_np[i, idx * self.k_shot])
			# pred: [b, querysz]
			pred = Variable(torch.from_numpy(np.array(pred).reshape((batchsz, querysz)))).cuda()

			correct = torch.eq(pred, query_y).sum()
			return pred, correct