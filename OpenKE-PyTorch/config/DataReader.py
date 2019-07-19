from .utils.Reader import _importTrainFiles, _importTestFiles, _importTypeFiles, _importContextFiles
from .utils.Base import _sampling
from .utils.Random import _randReset
from .utils.Valid import _validInit, _getValidHeadBatch, _getValidTailBatch, _validHead, _validTail, _getValidHit10
from .utils.Test import _initTest, _getHeadBatch, _getTailBatch, _testHead, _testTail, _test_link_prediction, _getNegTest, _getNegValid, _getTestBatch, _getValidBatch, _getBestThreshold, _test_triple_classification
import numpy as np

class ConArg(object):

	def __init__(self):
		self.inPath = ''
		# self.inPath = '/Users/zhangjiatao/Nutstore Files/我的坚果云/Project/OpenKE/benchmarks/FB13/' 
		self.outPath = ''
		self.workThreads = 1
		self.negEnt = 1
		self.negRel = 0
		self.bernFlag = 1
		self.batchSize = 10


class InData(object):

	def __init__(self):

		# ------- 总体统计 ------- 
		self.relationTotal = 0
		self.entityTotal = 0
		self.freqEnt = []
		self.freqRel = []

		# ------- 训练数据部分 ------- 
		self.trainList = [] # 存储train2id文件
		self.trainTotal = 0

		self.trainHead = [] # 去重后训练集 以head优先排序
		self.lefHead = [] # 对于trainHead列表建立的索引，地i个实体可以通过lefHead[i]获取到在该列表中以第i个为头实体的三元组的起点索引，下同
		self.rigHead = []

		self.trainTail = [] # 去重后训练集 以tail优先排
		self.lefTail = []
		self.rigTail = []

		self.trainRel = [] 
		self.lefRel = []
		self.rigRel = []

		self.left_mean = []
		self.right_mean = []

		self.neighbor_context = {}
		self.path_context = {}

		# ------- 测试及验证数据部分 ------- 

		# Reader部分
		self.testTotal = 0
		self.validTotal = 0 
		self.tripleTotal = 0

		self.testList = []
		self.testLef = []
		self.testRig = []

		self.validList = []
		self.validLef = []
		self.validRig = []

		self.tripleList = []



		# Valid部分
		self.lastValidHead = 0
		self.lastValidTail = 0
			
		self.l_valid_filter_tot = 0
		self.r_valid_filter_tot = 0

		self.validHit10 = 0


		# Test部分
		self.lastHead = 0
		self.lastTail = 0

		self.l_tot = 0
		self.l_filter_rank = 0
		self.l_rank = 0
		self.l_filter_reci_rank = 0
		self.l_reci_rank = 0
		self.l_filter_tot = 0
		self.l_tot_constrain = 0
		self.l_filter_rank_constrain = 0
		self.l_rank_constrain = 0
		self.l_filter_reci_rank_constrain = 0
		self.l_reci_rank_constrain = 0
		self.l_filter_tot_constrain = 0

		self.l1_filter_tot_constrain = 0
		self.l1_tot_constrain = 0
		self.l1_filter_tot = 0 
		self.l1_tot = 0

		self.l3_filter_tot_constrain = 0 
		self.l3_tot_constrain = 0
		self.l3_filter_tot = 0
		self.l3_tot = 0

		self.r_tot = 0
		self.r_filter_tot = 0
		self.r_filter_rank = 0
		self.r_rank = 0
		self.r_filter_reci_rank = 0
		self.r_reci_rank = 0 
		self.r_tot_constrain = 0 
		self.r_filter_tot_constrain = 0
		self.r_filter_rank_constrain = 0
		self.r_rank_constrain = 0
		self.r_filter_reci_rank_constrain = 0
		self.r_reci_rank_constrain = 0

		self.r1_tot = 0 
		self.r1_filter_tot = 0
		self.r1_tot_constrain = 0
		self.r1_filter_tot_constrain = 0

		self.r3_tot_constrain = 0
		self.r3_filter_tot_constrain = 0
		self.r3_tot = 0
		self.r3_filter_tot = 0

		self.negValidList = []
		self.negTestList = []

		self.testAcc = []
		self.aveAcc = 0
		self.relThresh = np.zeros((1, 1))

		# ------- 类型约束部分数据 ------- 
		self.head_lef = []
		self.head_rig = []
		self.tail_lef = []
		self.tail_rig = []
		self.tail_type = []
		self.total_lef = [] 
		self.total_rig = []

		# ------- 随机打乱部分 ------- 
		self.next_random = []


class DataReader(object):

	def __init__(self):

		self.conArg = ConArg()
		self.inData = InData()

	# set部分
	def setInPath(self, inPath):
		self.conArg.inPath = inPath

	def setOutPath(self, outPath):
		self.conArg.outPath = outPath

	def setNegEnt(self, negEnt):
		self.conArg.negEnt = negEnt

	def setNegRel(self, negRel):
		self.conArg.negRel = negRel

	def setBern(self, bernFlag):
		self.conArg.bernFlag = bernFlag

	def setBatchSize(self, batchSize):
		self.conArg.batchSize = batchSize

	# get部分
	def getRelationTotal(self):
		return self.inData.relationTotal

	def getEntityTotal(self):
		return self.inData.entityTotal

	def getTripleTotal(self):
		return self.inData.tripleTotal


	def getTrainTotal(self):
		return self.inData.trainTotal

	def getTestTotal(self):
		return self.inData.testTotal

	def getValidTotal(self):
		return self.inData.validTotal

	# 基础数据读取部分
	def randReset(self):
		_randReset(self.conArg, self.inData)

	def importTrainFiles(self):
		_importTrainFiles(self.conArg, self.inData)

	def importTestFiles(self):
		_importTestFiles(self.conArg, self.inData)

	def importTypeFiles(self):
		_importTypeFiles(self.conArg, self.inData)

	def importContextFiles(self):
		_importContextFiles(self.conArg, self.inData)

	# train部分
	def sampling(self):
		(self.batch_h,
		 self.batch_t,
		 self.batch_r,
		 self.batch_y) = _sampling(self.conArg, self.inData)

	# valid部分
	def validInit(self):
		_validInit(self.inData)

	def getValidHeadBatch(self):
		(self.valid_h,
		 self.valid_t,
		 self.valid_r) = _getValidHeadBatch(self.inData)

	def getValidTailBatch(self):
		(self.valid_h,
		 self.valid_t,
		 self.valid_r) = _getValidTailBatch(self.inData)

	def validHead(self, con):
		_validHead(con, self.inData)

	def validTail(self, con):
		_validTail(con, self.inData)

	def getValidHit10(self):
		_getValidHit10(self.inData)
		return self.inData.validHit10

	# test部分

	def initTest(self):
		_initTest()

	def getHeadBatch(self):
		(self.test_h,
		 self.test_t,
		 self.test_r) = _getHeadBatch(self.inData)

	def getTailBatch(self):
		(self.test_h,
		 self.test_t,
		 self.test_r) = _getTailBatch(self.inData)

	def testHead(self, con):
		_testHead(con, self.inData)

	def testTail(self, con):
		return _testTail(con, self.inData)

	def test_link_prediction(self):
		_test_link_prediction(self.inData)

	def getNegTest(inData):
		_getNegTest(self.inData)

	def getNegValid(inData):
		_getNegValid(self.inData)

	def getTestBatch(self):
		(self.test_pos_h,
		 self.test_pos_t,
		 self.test_pos_r,
		 self.test_neg_h,
		 self.test_neg_t,
		 self.test_neg_r) = _getTestBatch(self.inData)

	def getValidBatch(self):
		(self.valid_pos_h,
		 self.valid_pos_t,
		 self.valid_pos_r,
		 self.valid_neg_h,
		 self.valid_neg_t,
		 self.valid_neg_r) = _getValidBatch(self.inData)


	def getBestThreshold(self, score_pos, score_neg):
		_getBestThreshold(score_pos, score_neg, self.inData)


	def test_triple_classification(self, score_pos, score_neg):
		_test_triple_classification(score_pos, score_neg, self.inData)


	# 被砍掉的多线程
	# def setWorkThreads(self, workThreads):
	# 	self.conArg.workThreads = workThreads

def main():
	'''
	用于测试
	'''
	DR = DataReader()

	DR.importTrainFiles()
	print(len(DR.inData.trainList))

	DR.importContextFiles()
	print('neighbor',DR.inData.neighbor_context)
	print('path', DR.inData.path_context)

	# DR.importTestFiles()
	# print(len(DR.inData.testList))

	# DR.importTypeFiles()
	# print(DR.inData.total_lef)

	# DR.randReset()
	# print(DR.inData.next_random)

	# DR.sampling()

	# print(DR.batch_h)

	# DR.validInit()
	# DR.getValidHeadBatch()
	# print(DR.valid_t)

	# DR.getValidTailBatch()
	# print(DR.valid_h)


if __name__ == '__main__':
	main()
