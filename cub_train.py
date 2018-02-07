import torch, os
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch import  nn

from zeroshot import Zeroshot
from cub2 import Cub



# save best acc info, to save the best model to ckpt.
best_accuracy = 0
def evaluation(net, batchsz, n_way, mdl_file):
	"""
	obey the expriment setting of MAML and Learning2Compare, we randomly sample 600 episodes and 15 query images per query
	set.
	:param net:
	:param batchsz:
	:return:
	"""
	# we need to test 11788 - 8855 = 2933 images.
	k_query = 1
	db = Cub('../CUB_data/', n_way, k_query, train=False, episode_num= 3000//n_way//k_query)
	db_loader = DataLoader(db, batchsz, shuffle=True, num_workers=2, pin_memory=True)

	accs = []
	for batch in db_loader:
		x = Variable(batch[0]).cuda()
		x_label = Variable(batch[1]).cuda()
		att = Variable(batch[2]).cuda()
		att_label = Variable(batch[3]).cuda()


		pred, correct = net(x, x_label, att, att_label, False)
		correct = correct.sum().data[0] # multi-gpu

		# preds = torch.cat(preds, dim= 1)
		acc = correct / ( x_label.size(0) * x_label.size(1) )
		accs.append(acc)

	print(pred.cpu().data[0].numpy())
	print(accs)

	# compute the distribution of 600/episodesz episodes acc.
	global best_accuracy
	accuracy = np.array(accs).mean()
	print('<<<<<<<<< accuracy:', accuracy, 'best accuracy:', best_accuracy, '>>>>>>>>')

	if accuracy > best_accuracy:
		best_accuracy = accuracy
		torch.save(net.state_dict(), mdl_file)
		print('Saved to checkpoint:', mdl_file)

	return accuracy



def main():

	batchsz = 40
	n_way = 50
	k_query = 5
	lr = 1e-5
	mdl_file = 'ckpt/cub.mdl'


	net = nn.DataParallel(Zeroshot(), device_ids=[0]).cuda()
	print(net)

	if os.path.exists(mdl_file):
		print('load from checkpoint ...', mdl_file)
		net.load_state_dict(torch.load(mdl_file))
	else:
		print('training from scratch.')

	# whole parameters number
	model_parameters = filter(lambda p: p.requires_grad, net.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total params  :', params)

	optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=25, verbose=True)

	for epoch in range(1000):
		db = Cub('../CUB_data/', n_way, k_query, train=True)
		db_loader = DataLoader(db, batchsz, shuffle=True, num_workers=2, pin_memory=True)
		total_train_loss = 0

		for step, batch in enumerate(db_loader):

			# 1. test
			if step % 400 == 0:
				accuracy = evaluation(net, batchsz, n_way, mdl_file)
				scheduler.step(accuracy)

			# 2. train
			x = Variable(batch[0]).cuda()
			x_label = Variable(batch[1]).cuda()
			att = Variable(batch[2]).cuda()
			att_label = Variable(batch[3]).cuda()

			net.train()
			loss = net(x, x_label, att, att_label)
			loss = loss.sum() / (x_label.size(0) * x_label.size(1) ) # multi-gpu, divide by total batchsz
			total_train_loss += loss.data[0]

			optimizer.zero_grad()
			loss.backward()
			# nn.utils.clip_grad_norm(net.parameters(), 10)
			optimizer.step()

			# 3. print
			if step % 20 == 0 and step != 0:
				print('%d-way %d batch> epoch:%d step:%d, loss:%f' % (
				n_way,  batchsz, epoch, step, total_train_loss) )
				total_train_loss = 0



if __name__ == '__main__':
	main()
