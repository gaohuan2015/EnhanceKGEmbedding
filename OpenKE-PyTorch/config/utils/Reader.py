import json

def sort_head(tripleList):
	tripleList.sort(key = lambda k : (k.get('h', 0), k.get('r', 0), k.get('t', 0)))
	return tripleList

def sort_tail(tripleList):
	tripleList.sort(key = lambda k : (k.get('t', 0), k.get('r', 0), k.get('h', 0)))
	return tripleList

def sort_rel(tripleList):
	tripleList.sort(key = lambda k : (k.get('h', 0), k.get('t', 0), k.get('r', 0)))
	return tripleList

def sort_rel2(tripleList):
	tripleList.sort(key = lambda k : (k.get('r', 0), k.get('h', 0), k.get('t', 0)))
	return tripleList


def _importTrainFiles(conArg, inData):
	'''
	读入训练数据

	测试情况：跑通，正确性有待验证
	'''

	print("PY: The toolkit is importing datasets.")

	inPath = conArg.inPath
	# ============ 需要读取的数据 ============ 
	trainList = [] # 存储train2id文件
	trainHead = [] # 去重后训练集 以head优先排序
	trainTail = [] # 去重后训练集 以tail优先排序
	trainRel = [] # 去重后训练集 以r优先排序
	# 统计
	relationTotal = -1
	entityTotal = -1

	# 读取关系数量
	with open(inPath + "relation2id.txt", 'r') as fin:
		relationTotal = int(fin.readline())
		print('The total of relations is', relationTotal)
	fin.close()

	# 读取实体数量
	with open(inPath + "entity2id.txt", 'r') as fin:
		entityTotal = int(fin.readline())
		print('The total of entities is', entityTotal)
	fin.close()


	freqRel = [0] * relationTotal  # 统计在训练集中每个关联出现次数
	freqEnt = [0] * entityTotal # 统计在训练集中每个实体出现次数
	# 建立head、tail、rel列表的索引
	lefHead = [-1] * entityTotal # 对于trainHead列表建立的索引，地i个实体可以通过lefHead[i]获取到在该列表中以第i个为头实体的三元组的起点索引，下同
	rigHead = [-1] * entityTotal
	lefTail = [-1] * entityTotal
	rigTail = [-1] * entityTotal
	lefRel = [-1] * entityTotal
	rigRel = [-1] * entityTotal

	# 读取训练集
	with open(inPath + "train2id.txt", 'r') as fin:
		for index, line in enumerate(fin.readlines()):
			if index == 0:
				trainTotal = int(line)
			else:
				item = {}
				tmpList = line.replace('\n', '').split()
				item['h'] = int(tmpList[0])
				item['t'] = int(tmpList[1])
				item['r'] = int(tmpList[2])
				trainList.append(item)
		print('The trainList size is', trainTotal)
	fin.close()

	# 去重 and 重新封装统计
	trainList = sort_head(trainList) 
	trainHead.append(trainList[0])
	trainTail.append(trainList[0])
	trainRel.append(trainList[0])

	freqEnt[trainList[0]['h']] = 1
	freqEnt[trainList[0]['t']] = 1
	freqRel[trainList[0]['r']] = 1
	for i in range(1, len(trainList)):
		if ((trainList[i]['h'] != trainList[i - 1]['h']) or (trainList[i]['r'] != trainList[i - 1]['r']) or (trainList[i]['t'] != trainList[i - 1]['t'])):
			
			trainHead.append(trainList[i])
			trainTail.append(trainList[i])
			trainRel.append(trainList[i])
			freqEnt[trainList[i]['h']] += 1
			freqEnt[trainList[i]['t']] += 1
			freqRel[trainList[i]['r']] += 1

	trainHead = sort_head(trainHead)
	trainTail = sort_tail(trainTail)
	trainRel = sort_rel(trainRel)

	trainTotal = len(trainHead)
	print("The total of train triples is", trainTotal)


	# 计算在每种排序下的索引范围
	for i in range(trainTotal):
		if trainTail[i]['t'] != trainTail[i - 1]['t']:
			rigTail[trainTail[i - 1]['t']] = i - 1
			lefTail[trainTail[i]['t']] = i

		if trainHead[i]['h'] != trainHead[i - 1]['h']:
			rigHead[trainHead[i - 1]['h']] = i - 1
			lefHead[trainHead[i]['h']] = i

		if trainRel[i]['h'] != trainRel[i - 1]['h']:
			rigRel[trainRel[i - 1]['h']] = i - 1
			lefRel[trainRel[i]['h']] = i


	# 对最后一个三元组做边界处理
	lefHead[trainHead[0]['h']] = 0
	rigHead[trainHead[trainTotal - 1]['h']] = trainTotal - 1
	lefTail[trainTail[0]['t']] = 0
	rigTail[trainTail[trainTotal - 1]['t']] = trainTotal - 1
	lefRel[trainRel[0]['h']] = 0
	rigRel[trainRel[trainTotal - 1]['h']] = trainTotal - 1

	# print('ent id is', trainHead[0]['h'])

	# # 数据观察
	# for index in range(trainTotal):
	# 	if index >= 50:
	# 		break
	# 	else:
	# 		print(trainHead[index])

	# 这个mean到底是在算什么玩意？
	left_mean = [0] * relationTotal
	right_mean = [0] * relationTotal
	
	for i in range(entityTotal):

		for j in range(lefHead[i] + 1,  rigHead[i] + 1):
			if trainHead[j]['r'] != trainHead[j - 1]['r']:
				left_mean[trainHead[j]['r']] += 1.0
		if lefHead[i] <= rigHead[i]:
				left_mean[trainHead[lefHead[i]]['r']] += 1.0

		for j in range(lefTail[i] + 1, rigTail[i] + 1):
			if trainTail[j]['r'] != trainTail[j - 1]['r']:
				right_mean[trainTail[j]['r']] += 1.0

		if lefTail[i] <= rigTail[i]:
			right_mean[trainTail[lefTail[i]]['r']] += 1.0
		
  
	for i in range(relationTotal):
		left_mean[i] = freqRel[i] / left_mean[i]
		right_mean[i] = freqRel[i] / right_mean[i]

		inData.relationTotal, inData.entityTotal = relationTotal, entityTotal
		inData.freqRel, inData.freqEnt = freqRel, freqEnt
		inData.trainList, inData.trainTotal = trainList, trainTotal
		inData.trainHead, inData.lefHead, inData.rigHead = trainHead, lefHead, rigHead
		inData.trainTail, inData.lefRel, inData.rigRel = trainTail, lefRel, rigRel 
		inData.trainRel, inData.lefTail, inData.rigTail = trainTail, lefTail, rigTail
		inData.left_mean, inData.right_mean = left_mean, right_mean 






def _importTestFiles(conArg, inData):
	'''
	测试数据读入

	测试情况：跑通， 正确性有待验证
	'''
	inPath = conArg.inPath

	# 读取关系类型数量
	with open(inPath + "relation2id.txt", 'r') as fin:
		relationTotal = int(fin.readline())
		print('The total of relations is', relationTotal)
	fin.close()

	# 读取实体数量
	with open(inPath + "entity2id.txt", 'r') as fin:
		entityTotal = int(fin.readline())
		print('The total of entities is', entityTotal)
	fin.close()


	# 读取 
	testList = []
	validList = []
	tripleList = []  # TODO tripleList的len和tripleTotal不一样，有bug需要解决

	# 读取测试集
	with open(inPath + "test2id.txt", 'r') as fin:
		for index, line in enumerate(fin.readlines()):
			if index == 0:
				testTotal = int(line)
			else:
				item = {}
				tmpList = line.replace('\n', '').split()
				item['h'] = int(tmpList[0])
				item['t'] = int(tmpList[1])
				item['r'] = int(tmpList[2])
				testList.append(item)
				tripleList.append(item)
		print('The testList size is', testTotal)
	fin.close()

	# 读取训练集
	with open(inPath + "train2id.txt", 'r') as fin:
		for index, line in enumerate(fin.readlines()):
			if index == 0:
				trainTotal = int(line)
			else:
				item = {}
				tmpList = line.replace('\n', '').split()
				item['h'] = int(tmpList[0])
				item['t'] = int(tmpList[1])
				item['r'] = int(tmpList[2])
				tripleList.append(item)
	fin.close()

	# 读取验证
	with open(inPath + "valid2id.txt", 'r') as fin:
		for index, line in enumerate(fin.readlines()):
			if index == 0:
				validTotal = int(line)
			else:
				item = {}
				tmpList = line.replace('\n', '').split()
				item['h'] = int(tmpList[0])
				item['t'] = int(tmpList[1])
				item['r'] = int(tmpList[2])
				validList.append(item)
				tripleList.append(item)
	fin.close()
	tripleTotal = testTotal + trainTotal + validTotal;

	# print('trainTotal', inData.trainTotal)
	# print('trainList', len(inData.trainList))

	# print('testTotal', testTotal)
	# print('testList', len(testList))

	# print('validTotal', validTotal)
	# print('validList', len(validList))

	# print('tripleTotal', tripleTotal)
	# print('tripleList', len(tripleList))

	tripleList = sort_head(tripleList)
	testList = sort_rel2(testList)
	validList = sort_rel2(validList)

	print("The total of test triples is", testTotal);
	print("The total of valid triples is", validTotal);

	testLef = [-1] * relationTotal # 初始化为-1
	testRig = [-1] * relationTotal # 初始化为-1


	for i in range(1, testTotal):
		if testList[i]['r'] != testList[i-1]['r']:
			testRig[testList[i-1]['r']] = i - 1;
			testLef[testList[i]['r']] = i;
	
	testLef[testList[0]['r']] = 0;
	testRig[testList[testTotal - 1]['r']] = testTotal - 1;

	validLef = [-1] * relationTotal
	validRig = [-1] * relationTotal

	for i in range(1, validTotal):
		if validList[i]['r'] != validList[i-1]['r']:
			validRig[validList[i-1]['r']] = i - 1;
			validLef[validList[i]['r']] = i;

	validLef[validList[0]['r']] = 0;
	validRig[validList[validTotal - 1]['r']] = validTotal - 1;


	inData.relationTotal, inData.entityTotal = relationTotal, entityTotal
	inData.trainTotal, inData.testTotal, inData.validTotal, inData.tripleTotal =  trainTotal, testTotal, validTotal, tripleTotal
	inData.testList, inData.testLef, inData.testRig = testList, testLef, testRig 
	inData.validList, inData.validLef, inData.validRig = validList, validLef, validRig
	inData.tripleList = tripleList



def _importTypeFiles(conArg, inData):
	'''
	读取关系约束数据

	测试情况：
	'''
	inPath = conArg.inPath
	relationTotal = inData.relationTotal

	head_lef = [-1] * relationTotal
	head_rig = [-1] * relationTotal
	tail_lef = [-1] * relationTotal
	tail_rig = [-1] * relationTotal
	total_lef = 0
	total_rig = 0
	tmp = 0

	# 第一遍读取，获取total_lef, total_rig
	with open(inPath + "type_constrain.txt", 'r') as fin:
		
		tmp = int(fin.readline()) # 读取限制数量，一般和relationTotal数量保持一致
		print(relationTotal)

		# 对每个关系获取其头实体数量和尾实体数量
		for i in range(relationTotal):

			line =  fin.readline()
			tmpList = [int(k) for k in line.replace('\n', '').split()] # 读一行
			rel = tmpList[0] # 关系id
			tot = tmpList[1] # 头实体约束数量
			for j in range(tot):
				tmp = tmpList[j + 1]
				total_lef += 1

			line =  fin.readline()
			tmpList = [int(k) for k in line.replace('\n', '').split()] # 再读一行
			rel = tmpList[0] # 关系id
			tot = tmpList[1] # 尾实体约束数量
			for j in range(tot):
				tmp = tmpList[j + 1]
				total_rig += 1
	fin.close()

	head_type = []
	tail_type = []
	# head_type = [-1] * total_lef # 建立类型索引开始部分
	# tail_type = [-1] * total_rig # 建立类型索引结束部分

	# print('total_lef and total_rig',total_lef, total_rig)

	total_lef = 0
	total_rig = 0

	# 第二遍读取
	with open(inPath + "type_constrain.txt", 'r') as fin:
		
		tmp = int(fin.readline()) # 读取限制数量，应该是与relationTotal数量保持一致

		for i in range(relationTotal):

			tmp_head_type = [] # 暂存数据，用于排序
			tmp_tail_type = [] # 暂存数据，用于排序

			line = fin.readline()
			tmpList = [int(k) for k in line.replace('\n', '').split()] # 读一行
			rel = tmpList[0]
			tot = tmpList[1]
			head_lef[rel] = total_lef # 记录该关系头实约束起点
			for j in range(tot):
				tmp = tmpList[j + 1]
				tmp_head_type.append(tmp)
				total_lef += 1

			head_rig[rel] = total_lef # 记录该关系头实体约束终点

			# 对临时列表进行排序后将其再装入原列表
			tmp_head_type.sort()
			head_type = head_type + tmp_head_type

			line = fin.readline()
			tmpList = [int(k) for k in line.replace('\n', '').split()] # 再读一行
			rel = tmpList[0]
			tot = tmpList[1]
			tail_lef[rel] = total_rig
			for j in range(tot):
				tmp = tmpList[j + 1]
				tmp_tail_type.append(tmp)
				total_rig += 1

			tail_rig[rel] = total_rig

			# 对临时列表排序后再装回来
			tmp_tail_type.sort()
			tail_type = tail_type + tmp_tail_type
			# print('rel, len:', rel, tail_rig[rel] - tail_lef[rel])
	fin.close()

	# # 打印第0个关系的尾实体
	# print('tail_type 0, the len is')
	# print(tail_rig[0] - tail_lef[0])
	# for i in range(tail_lef[0], tail_rig[0]):
	# 	print(tail_type[i])

	inData.head_type, inData.tail_type = head_type, tail_type
	inData.total_lef, inData.total_rig = total_lef, total_rig 
	inData.head_lef, inData.head_rig = head_lef, head_rig
	inData.tail_lef, inData.tail_rig = tail_lef, tail_rig
	

def _importContextFiles(conArg, inData):
	'''
	读入上下文信息
	'''
	inPath = conArg.inPath
	relationTotal = inData.relationTotal
	entityTotal = inData.entityTotal

	# 读取neighbor context信息
	with open(inPath + "neighbor_context.txt", 'r') as fin:
		tmp_str = fin.readline()
		neighbor_context = json.loads(tmp_str)
	fin.close()

	# 读取path context信息
	with open(inPath + "path_context.txt", 'r') as fin:
		tmp_str = fin.readline()
		path_context = json.loads(tmp_str)
	fin.close()

	inData.neighbor_context = neighbor_context
	inData.path_context = path_context


def main():
	'''
	用于对当前脚本进行测试
	'''
	# print('read TrainFile')
	# _importTrainFiles()
	# print(len(trainList))


	# print('read ContextFile')
	# )

	# print('Running Reader....')
	# importTestFiles()
	# print(len(trainList))
	return 

if __name__ == '__main__':
	main()