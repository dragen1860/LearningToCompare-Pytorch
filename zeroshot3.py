from scipy import io
import numpy as np
import torch, sys
from torch.autograd import Variable
from torch.nn import functional as F
import kNN
from cub_bad import Cub
from torch.utils.data import DataLoader

# def compute_accuracy(test_att, test_visual, test_id, test_label):
# 	# test_att: [2993, 312]
# 	# test viaual: [2993, 1024]
# 	# test_id: att2label [50]
# 	# test_label: x2label [2993]
# 	test_att = Variable(torch.from_numpy(test_att).float().cuda())
# 	att_pred = forward(test_att)
# 	outpred = [0] * 2933
# 	test_label = test_label.astype("float32")
#
# 	# att_pre [50, 1024],
# 	# test_visual: [2993, 1024]
# 	# test_id : [50]
#
# 	for i in range(2933):
# 		outputLabel = kNN.kNNClassify(test_visual[i, :], att_pred.cpu().data.numpy(), test_id, 1)
# 		outpred[i] = outputLabel
# 	outpred = np.array(outpred)
# 	acc = np.equal(outpred, test_label).mean()
#
# 	return acc


# def data_iterator(batch_size):
#
# 	while True:
# 		# shuffle labels and features
# 		idxs = np.arange(0, len(train_x))
# 		np.random.shuffle(idxs)
# 		shuf_visual = train_x[idxs]
# 		shuf_att = train_att[idxs]
#
# 		for batch_idx in range(0, len(train_x), batch_size):
# 			visual_batch = shuf_visual[batch_idx:batch_idx + batch_size].astype(np.float32)
# 			att_batch = shuf_att[batch_idx:batch_idx + batch_size]
#
# 			att_batch = Variable(torch.from_numpy(att_batch).float().cuda())
# 			visual_batch = Variable(torch.from_numpy(visual_batch).float().cuda())
#
# 			yield att_batch, visual_batch


f = io.loadmat('../CUB_data/train_attr.mat')
train_att = np.array(f['train_attr'])
print('train attr:', train_att.shape)

f = io.loadmat('../CUB_data/train_cub_googlenet_bn.mat')
train_x = np.array(f['train_cub_googlenet_bn'])
print('train x:', train_x.shape)

f = io.loadmat('../CUB_data/test_cub_googlenet_bn.mat')
test_x = np.array(f['test_cub_googlenet_bn'])
print('test x:', test_x.shape)

f = io.loadmat('../CUB_data/test_proto.mat')
test_att = np.array(f['test_proto'])
print('test att:', test_att.shape)

f = io.loadmat('../CUB_data/test_labels_cub.mat')
test_x2label = np.squeeze(np.array(f['test_labels_cub']))
print('test x2label:', test_x2label)

f = io.loadmat('../CUB_data/testclasses_id.mat')
test_att2label = np.squeeze(np.array(f['testclasses_id']))
print('test att2label:', test_att2label)

w1 = Variable(torch.FloatTensor(312, 700).cuda(), requires_grad=True)
b1 = Variable(torch.FloatTensor(700).cuda(), requires_grad=True)
w2 = Variable(torch.FloatTensor(700, 1024).cuda(), requires_grad=True)
b2 = Variable(torch.FloatTensor(1024).cuda(), requires_grad=True)

# must initialize!
w1.data.normal_(0, 0.02)
w2.data.normal_(0, 0.02)
b1.data.fill_(0)
b2.data.fill_(0)


def forward(att):
	a1 = F.relu(torch.mm(att, w1) + b1)
	a2 = F.relu(torch.mm(a1, w2) + b2)

	return a2


def getloss(pred, x):
	loss = torch.pow(x - pred, 2).sum()
	loss /= x.size(0)
	return loss


optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=1e-5, weight_decay=1e-2)

n_way = 50
batchsz = 1
db = Cub('../CUB_data/', n_way, 1, train=True)
db_loader = DataLoader(db, batchsz, shuffle=True, num_workers=2, pin_memory=True)

# db_iter = data_iterator(n_way * batchsz)

# np.random.seed(666)


# def get_att_from_iter(x, att):
# 	# x [b, setsz, 1024]
# 	batchsz, setsz, c = x.size()
# 	x = x.view(-1, 1024).cpu().data.numpy()
# 	att = att.view(-1, 312).cpu().data.numpy()
# 	buff = []
# 	for vec in x:
# 		x_in_iter_idx = np.where((train_x == vec).all(axis=1))[0][0]
# 		att_in_iter = train_att[x_in_iter_idx]
# 		x_in_iter = train_x[x_in_iter_idx]
# 		assert np.isclose(x, x_in_iter).all()
# 		assert np.isclose(att, att_in_iter).all()
#
# 		buff.append(att_in_iter)
# 	buff = Variable(torch.from_numpy(np.array(buff).astype(np.float32).reshape(batchsz, setsz, 312)).cuda())
# 	return buff




for step, batch in enumerate(db_loader):
	x_b = batch[0][0]
	att_b = batch[2][0]
	x_label_b = batch[1][0]
	att_label_b = batch[3][0]

	assert torch.equal(x_label_b, att_label_b)

	for i, (x, att) in enumerate(zip(x_b, att_b)):
		x = x.numpy()
		att = att.numpy()
		x_in_iter_idx = np.where((train_x == x).all(axis=1))[0][0]
		att_in_iter = train_att[x_in_iter_idx]
		x_in_iter = train_x[x_in_iter_idx]
		assert np.isclose(x, x_in_iter).all()
		assert np.isclose(att, att_in_iter).all()
	if step > 10:
		break
print('data match!')

while True:

	for step, batch in enumerate(db_loader):
		x = Variable(batch[0]).cuda()
		x_label = Variable(batch[1]).cuda()
		att = Variable(batch[2]).cuda()
		att_label = Variable(batch[3]).cuda()
		# x [b, setsz, c]
		# x_label [b, setsz]
		# att [b, n_way, 312]
		# att_label [b, n_way]


		att = att.view(-1, att.size(2))
		x= x.view(-1, x.size(2))

		pred = forward(att)
		loss = getloss(pred, x)

		optimizer.zero_grad()
		loss.backward()
		# gradient clip makes it converge much faster!
		# for p in [w1, b1, w2, b2]:
		# 	print(p.grad.norm(2).data[0])
		torch.nn.utils.clip_grad_norm([w1, b1, w2, b2], 1)
		optimizer.step()


		if step % 1000 == 0:
			print(loss.data[0])
			# print(compute_accuracy(test_att, test_x, test_att2label, test_x2label), 'loss:', loss.data[0])
			# print(pred[0], visual_batch_val[0])
