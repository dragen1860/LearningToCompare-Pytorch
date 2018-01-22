import torch
import numpy as np
from torch.autograd import Variable


def make_imgs(n_way, k_shot, n_query_per_cls, batchsz,
              support_x, support_y, query_x, query_y, query_pred,
              mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
	# randomly select one batch
	batchidx = np.random.randint(batchsz)
	# k_shot . n_qeuery_per_cls...6... n_query_per_cls
	max_width = (k_shot + n_query_per_cls * 2 + 4)

	# de-normalize
	img_support = support_x[batchidx].clone()
	img_query = query_x[batchidx].clone()
	img_support = img_support * Variable(torch.FloatTensor(list(std)).view(1, 3, 1, 1)).cuda() + Variable(
		torch.FloatTensor(list(mean)).view(1, 3, 1, 1)).cuda()
	img_query = img_query * Variable(torch.FloatTensor(list(std)).view(1, 3, 1, 1)).cuda() + Variable(
		torch.FloatTensor(list(mean)).view(1, 3, 1, 1)).cuda()

	label = support_y[batchidx]  # [setsz]
	label, indices = torch.sort(label, dim=0)
	# [setsz, c, h, w] sort by indices
	img_support = torch.index_select(img_support, dim=0, index=indices)  # [setsz, c, h, w]
	all_img = torch.zeros(max_width * n_way, *img_support[0].size())  # [max_width * n_way, c, h, w]

	for row in range(n_way):  # for each row
		# [0, k_shot)
		for pos in range(k_shot):  # copy the first k_shot
			all_img[row * max_width + pos] = img_support[row * k_shot + pos].data

		# now set the pred imgs
		# [k_shot+1, max_width - n_query_per_cls -1]
		pos = k_shot + 1  # pointer to empty buff
		for idx, img in enumerate(img_query):  # search all imgs in pred that match current row id: label[row*k_shot]
			if torch.equal(query_pred[batchidx][idx], label[row * k_shot]):  # if pred it match current id
				if pos == max_width - n_query_per_cls:  # overwrite the last column
					pos -= 1
				all_img[row * max_width + pos] = img.data  # copy img
				pos += 1

		# set the last several column as the right img
		# [max_width - n_query_per_cls, max_width)
		pos = max_width - n_query_per_cls
		for idx, img in enumerate(img_query):  # search all imgs in pred that match current row id: label[row*k_shot]
			if torch.equal(query_y[batchidx][idx], label[row * k_shot]):  # if query_y id match current id
				if pos == max_width:  # overwrite the last column
					pos -= 1
				all_img[row * max_width + pos] = img.data  # copy img
				pos += 1

	print('label for support:', label.data.cpu().numpy().tolist())
	print('label for query  :', query_y.data[batchidx].cpu().numpy())
	print('label for pred   :', query_pred.data[batchidx].cpu().numpy())

	return all_img, max_width
