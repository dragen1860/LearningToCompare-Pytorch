import torch, os
import numpy as np
from torch.autograd import Variable
from torch import  nn



class Zeroshot(nn.Module):


	def __init__(self):
		super(Zeroshot, self).__init__()

		self.net = nn.Sequential(nn.Linear(312, 900),
		                         nn.ReLU(inplace=True),
		                         nn.Linear(900, 1024),
		                         nn.ReLU(inplace=True))

	def forward(self, x, x_label, att, att_label, train=True):
		"""

		:param x:           [b, setsz, c]
		:param x_label:     [b, setsz]
		:param att:         [b, n_way, 312]
		:param att_label:   [b, n_way]
		:return:
		"""
		batchsz, setsz, c = x.size()
		n_way = att.size(1)

		x = x.view(batchsz * setsz, c)
		att = att.view(batchsz * n_way, 312)
		att = self.net(att)

		if train:
			loss = torch.pow(x - att, 2).sum() / (batchsz * setsz)
			return loss

		else:
			x = x.view(batchsz, setsz, c)
			att = att.view(batchsz, n_way, 1024)

			x = x.unsqueeze(2).expand(batchsz, setsz, n_way, c)
			att = att.unsqueeze(1).expand(batchsz, setsz, n_way, c)
			# [b, setsz, n, c] => [b, setsz, n] => [b, setsz]
			_, indices = torch.pow(x - att, 2).sum(3).min(2)
			# [b, setsz]
			pred = torch.gather(att_label, 1, index=indices)
			correct = torch.eq(pred, x_label).sum()
			return pred, correct



def test():
	from cub2 import  Cub
	from torch import optim
	from torch.utils.data import DataLoader

	batchsz = 50
	n_way = 50
	k_query = 1
	lr = 1e-5
	mdl_file = 'ckpt/cub2.mdl'


	net = Zeroshot().cuda()

	optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-2)

	for epoch in range(1000):
		db = Cub('../CUB_data/', n_way, k_query, train=True)
		db_loader = DataLoader(db, batchsz, shuffle=True, num_workers=2, pin_memory=True)
		total_train_loss = 0

		for step, batch in enumerate(db_loader):
			# 2. train
			x = Variable(batch[0]).cuda()
			x_label = Variable(batch[1]).cuda()
			att = Variable(batch[2]).cuda()
			att_label = Variable(batch[3]).cuda()

			net.train()
			loss = net(x, x_label, att, att_label)
			total_train_loss += loss.data[0]

			optimizer.zero_grad()
			loss.backward()
			# if np.random.randint(1000)<4:
			# 	for p in net.parameters():
			# 		print(p.grad.norm(2).data[0])
			nn.utils.clip_grad_norm(net.parameters(), 1)
			optimizer.step()

			# 3. print
			if step % 20 == 0 and step != 0:
				print('%d-way %d batch> epoch:%d step:%d, loss:%f' % (
				n_way,  batchsz, epoch, step, loss.data[0]) )
				total_train_loss = 0



if __name__ == '__main__':
	test()