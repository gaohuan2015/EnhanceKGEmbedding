
from visdom import Visdom
import numpy as np
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib

threshold = 4000

def readFile(file):

	cnt_dict = {}
	item_list = []
	label_list = []
	index = 0
	cnt_pos = 0
	cnt_neg = 0
	with open(file, 'r') as f:
		f.readline()

		while(1):
			line = f.readline()
			if line == '':
				break

			item = line.replace('\n', '').split(' ')
			item = [float(i) for i in item]

			# 统计最多了hr
			key = str(item[0]) + ',' + str(item[1])

			if key != '101.0,91.0':
				continue



			if key not in cnt_dict:
				cnt_dict[key] = 1
			else:
				cnt_dict[key] += 1


			item_list.append(item)

			if item[3] >= threshold:
				label_list.append(1)
				cnt_neg += 1
			else:
				label_list.append(2)
				cnt_pos += 1

			index += 1

	f.close()

	print('[INFO] 正例', cnt_pos, '负例', cnt_neg)

	maximum = 0
	max_pair = ''
	for key in cnt_dict.keys():
		if cnt_dict[key] > maximum:
			maximum = cnt_dict[key]
			max_pair = key

	print('[INFO] 最多的hr是:', max_pair, '数量是：', maximum)

	return item_list, label_list


def draw(items, labels):

	length = len(items)

	viz = Visdom()
	assert viz.check_connection()

	Y = labels
	freq = items[:, 4]
	norm_F = items[:, 5]

	freq = freq.reshape((length, 1))
	norm_F = norm_F.reshape((length, 1))

	print('[INFO] draw : frew.shape', freq.shape)
	print('[INFO] draw : norm_F.shape', norm_F.shape)
	X = np.concatenate([freq, norm_F], 1)
	print('[INFO] draw : X.shape', X.shape)

	# scatter = viz.scatter(

	#     X=np.random.rand(100, 2),
	#     Y=(Y[Y > 0] + 1.5).astype(int),
	#     opts=dict(
	#         legend=['Apples', 'Pears'],
	#         xtickmin=0,
	#         xtickmax=1,
	#         xtickstep=0.5,
	#         ytickmin=0,
	#         ytickmax=1,
	#         ytickstep=0.5,
	#         markersymbol='cross-thin-open',

	#     ),
	# )

	# viz.scatter(
	#     X=np.random.rand(255, 2),
	#     #随机指定1或者2
	#     Y=(np.random.rand(255) + 1.5).astype(int),
	#     opts=dict(
	#         markersize=10,
	#         ## 分配两种颜色
	#         markercolor=np.random.randint(0, 255, (2, 3,)),
	#     ),
	# )

	#3D 散点图
	viz.scatter(
	    X=X,
	    Y=Y,
	    opts=dict(
	        legend=['NEG', 'POS'],
	        markersize=5,
	    )
	)


def main():
	item_list, label_list = readFile('/Users/zhangjiatao/Documents/EnhanceKGEmbedding/OpenKE-PyTorch/badCase.txt')
	draw(np.array(item_list), np.array(label_list, dtype = np.int32))
	return 


if __name__ == '__main__':
	main()



