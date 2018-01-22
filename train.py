import torch, os
import numpy as np
from torch import optim
from torch.autograd import Variable
from MiniImagenet import MiniImagenet
from compare import Compare
from utils import make_imgs

if __name__ == '__main__':
	from MiniImagenet import MiniImagenet
	from torch.utils.data import DataLoader
	from torchvision.utils import make_grid
	from tensorboardX import SummaryWriter
	from datetime import datetime

	n_way = 5
	k_shot = 5
	k_query = 1 # query num per class
	batchsz = 3
	# Multi-GPU support
	print('To run on single GPU, change device_ids=[0] and downsize batch size! \nmkdir ckpt if not exists!')
	net = torch.nn.DataParallel(Compare(n_way, k_shot), device_ids=[0]).cuda()
	# print(net)
	mdl_file = 'ckpt/compare%d%d.mdl'%(n_way, k_shot)

	if os.path.exists(mdl_file):
		print('load checkpoint ...', mdl_file)
		net.load_state_dict(torch.load(mdl_file))

	model_parameters = filter(lambda p: p.requires_grad, net.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('total params:', params)

	optimizer = optim.Adam(net.parameters(), lr=1e-3)
	tb = SummaryWriter('runs', str(datetime.now()))

	best_accuracy = 0
	for epoch in range(1000):

		mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query,
		                    batchsz=10000, resize=224)
		db = DataLoader(mini, batchsz, shuffle=True, num_workers=8, pin_memory=True)
		mini_val = MiniImagenet('../mini-imagenet/', mode='val', n_way=n_way, k_shot=k_shot, k_query=k_query,
		                        batchsz=200, resize=224)
		db_val = DataLoader(mini_val, batchsz, shuffle=True, num_workers=2, pin_memory=True)

		for step, batch in enumerate(db):
			support_x = Variable(batch[0]).cuda()
			support_y = Variable(batch[1]).cuda()
			query_x = Variable(batch[2]).cuda()
			query_y = Variable(batch[3]).cuda()

			net.train()
			loss = net(support_x, support_y, query_x, query_y)
			loss = loss.mean() # Multi-GPU support

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_val_loss = 0
			if step % 200 == 0:
				total_correct = 0
				total_num = 0
				display_onebatch = False # display one batch on tensorboard
				for j, batch_test in enumerate(db_val):
					support_x = Variable(batch_test[0]).cuda()
					support_y = Variable(batch_test[1]).cuda()
					query_x = Variable(batch_test[2]).cuda()
					query_y = Variable(batch_test[3]).cuda()

					net.eval()
					pred, correct = net(support_x, support_y, query_x, query_y, False)
					total_correct += correct.data[0]
					total_num += query_y.size(0) * query_y.size(1)

					if not display_onebatch:
						display_onebatch = True  # only display once
						all_img, max_width = make_imgs(n_way, k_shot, k_query, support_x.size(0),
						                               support_x, support_y, query_x, query_y, pred)
						all_img = make_grid(all_img, nrow=max_width)
						tb.add_image('result batch', all_img)

				accuracy = total_correct / total_num
				if accuracy > best_accuracy:
					best_accuracy = accuracy
					torch.save(net.state_dict(), mdl_file)
					print('saved to checkpoint:', mdl_file)

				tb.add_scalar('accuracy', accuracy)
				print('<<<<>>>>accuracy:', accuracy, 'best accuracy:', best_accuracy)

			if step % 15 == 0 and step != 0:
				tb.add_scalar('loss', loss.cpu().data[0])
				print('%d-way %d-shot %d batch> epoch:%d step:%d, loss:%f' % (
				n_way, k_shot, batchsz, epoch, step, loss.cpu().data[0]))
