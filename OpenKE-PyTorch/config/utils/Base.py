import numpy as np
from .Corrupt import corrupt_head, corrupt_tail, corrupt_rel, corrupt
from .Random import rand_max, randd
import time

def _sampling(conArg, inData):
	'''
	batch采样
	检查情况：对比检查通过
	
	采集到的batch结构说明：
	

	:param conArg: batch采样读取控制参数
	:param inData: 训练数据容器
	:return batch_h: 采集到的batch的头实体列表类型为np.array，其中元素类型为np.int64
	:return batch_t: batch尾实体
	:return batch_r: batch关系
	:return batch_y: batch标签(1是正例，0是负例)

	'''

	# 载入数据
	trainList, trainTotal = inData.trainList, inData.trainTotal
	relationTotal, entityTotal = inData.relationTotal, inData.entityTotal 
	freqRel, freqEnt = inData.freqRel, inData.freqEnt
	trainHead, lefHead, rigHead = inData.trainHead, inData.lefHead, inData.rigHead
	trainRel, lefRel, rigRel = inData.trainRel, inData.lefRel, inData.rigRel
	trainTail, lefTail, rigTail = inData.trainTail, inData.lefTail, inData.rigTail 
	right_mean, left_mean = inData.right_mean, inData.left_mean
	next_random = inData.next_random
	batchSize, workThreads, bernFlag = conArg.batchSize, conArg.workThreads, conArg.bernFlag
	negEnt, negRel = conArg.negEnt, conArg.negRel


	id = 0 # 单线程id恒为零
	batch_seq_size = batchSize * (1 + negEnt + negRel)
	batch_h = np.zeros(batch_seq_size, dtype=np.int64)
	batch_t = np.zeros(batch_seq_size, dtype=np.int64)
	batch_r = np.zeros(batch_seq_size, dtype=np.int64)
	batch_y = np.zeros(batch_seq_size, dtype=np.float32)

	prob = 500 # 这个变量作用是什么？

	for batch in range(batchSize):

		# 随机选择正例
		i = rand_max(next_random, id, trainTotal)
		batch_h[batch] = trainList[i]['h']
		batch_r[batch] = trainList[i]['r']
		batch_t[batch] = trainList[i]['t']
		batch_y[batch] = 1

		# 构造负例
		last = batchSize
		for times in range(negEnt):
			if bernFlag == 1:
				prob = 1000 * (1.0 * right_mean[trainList[i]['r']]) / (1.0 * right_mean[trainList[i]['r']] + 1.0 * left_mean[trainList[i]['r']])
			if (randd(next_random, id) % 1000 < prob):
				batch_h[batch + last] = trainList[i]['h']
				batch_t[batch + last] = corrupt_head(id, trainList[i]['h'], trainList[i]['r'], inData)
				batch_r[batch + last] = trainList[i]['r']
			else:
				batch_h[batch + last] = corrupt_tail(id, trainList[i]['t'], trainList[i]['r'], inData)
				batch_t[batch + last] = trainList[i]['t']
				batch_r[batch + last] = trainList[i]['r']
			
			batch_y[batch + last] = -1
			last += batchSize 
		
		for times in range(negRel):
			batch_h[batch + last] = trainList[i]['h']
			batch_t[batch + last] = trainList[i]['h']
			batch_r[batch + last] = corrupt_rel(id, trainList[i]['h'], trainList[i]['h'], lefRel, rigRel, trainRel, relationTotal, next_random)
			batch_y[batch + last] = -1
			last += batchSize


	return (batch_h, batch_t, batch_r, batch_y)

if __name__ == '__main__':
	main()
