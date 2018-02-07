import torch, os
import numpy as np
from torch.autograd import Variable
from torch import  nn

class Zeroshot(nn.Module):

	def __init__(self):
		super(Zeroshot, self).__init__()


		self.att_dim = 1024
		self.attnet = nn.Sequential(nn.Linear(312, 900),
		                            nn.ReLU(inplace=True),
		                            nn.Linear(900, self.att_dim),
		                            nn.ReLU(inplace=True))

		self.o = nn.Sequential(nn.Linear(self.att_dim * 2, 900),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(900, 128),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(128, 1))




	def forward(self, x, x_label, att, att_label, train = True):
		"""

		:param x:           [batchsz, setsz, c]
		:param x_label:     [batchsz, setsz]
		:param att:         [batchsz, n_way, 312]
		:param att_label:   [batchsz, n_way]
		:param train: for train or pred
		:return:
		"""

		batchsz, setsz, c = x.size()
		n_way = att.size(1)

		# make combination now
		# [b, setsz, c] => [b, setsz, 1, c] => [b, setsz, n_way, c]
		x_f = x.unsqueeze(2).expand(batchsz, setsz, n_way, c)
		# [b, n_way, 312] => [b, n_way, att_dim]
		att_f = self.attnet(att.view(batchsz * n_way, 312)).view(batchsz, n_way, self.att_dim)
		# => [b, 1, n_way, att_dim] => [b, setsz, n_way, att_dim]
		att_f = att_f.unsqueeze(1).expand(batchsz, setsz, n_way, self.att_dim)
		# [b, setsz, n_way, c + self.att_dim]
		comb = torch.cat([x_f, att_f], dim=3)
		c += self.att_dim # udpate c

		# [b, setsz, n_way]
		score = self.o(comb.view(batchsz * setsz * n_way, c)).view(batchsz, setsz, n_way)

		# build its label
		# [b, setsz] => [b, setsz, 1] => [b, setsz, n_way]
		x_labelf = x_label.unsqueeze(2).expand(batchsz, setsz, n_way)
		# [b, n_way] => [b, 1, n_way] => [b, setsz, n_way]
		att_labelf = att_label.unsqueeze(1).expand(batchsz, setsz, n_way)
		# eq: [b, setsz, n_way] => [b, setsz, n_way] and convert byte tensor to float tensor
		label = torch.eq(x_labelf, att_labelf).float()
		# print(label[0,0])
		# print(score[0,0])



		# score: [b, setsz, n_way]
		# label: [b, setsz, n_way]
		if train:
			loss = torch.pow(label - score, 2).sum()
			return loss

		else:
			# [b, setsz, n_way] => [b, setsz]
			_, indices = score.max(dim = 2)
			# att_label: [b, n_way]
			# indices: [b, setsz]
			# pred: [b, setsz], global true label
			pred = torch.gather(att_label, dim=1, index=indices)
			# print('scor:', score.cpu().data[0].numpy())
			# print('attl:', att_label.cpu().data[0].numpy())
			# print('pred:', pred.cpu().data[0].numpy())
			# print('x  l:', x_label.cpu().data[0].numpy())

			correct = torch.eq(pred, x_label).sum()
			return pred, correct





def test():
	net = Zeroshot()
	# whole parameters number
	model_parameters = filter(lambda p: p.requires_grad, net.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total params:', params)



if __name__ == '__main__':
	test()