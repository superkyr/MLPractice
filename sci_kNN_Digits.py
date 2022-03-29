from distutils.log import error
import numpy as np 
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

"""
def img2Vector(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    imgList = []
    
    for line in arrayOLines:
        line = line.strip()
        imgList.append(line)
    
    imgMat = np.array(imgList)
    imgVector = imgMat.reshape(1,-1)
    return imgVector
"""



"""
def classify0(inX, dataSet, labels, k):
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 在列向量方向上重复inX共1次（横向），行向量方向上重复inX共dataSetSize次（纵向）
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加，sum(axis=0)列相加，sum(axis=1)行相加
    sqDistances = sqDiffMat.sum(axis = 1)
    # 开方，计算出距离
    distances = sqDistances ** 0.5
    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None), 字典的get()方法，返回指定键的值，如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        # python3中用items()替换python2中的iteritems()
        # key=operator.itemgetter(1)根据字典的值进行排序
        # key=operator.itemgetter(0)根据字典的键进行排序
        # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别，即所要分类的类别
    return sortedClassCount[0][0]
"""

"""
函数说明：将32*32的二进制图像转换为1*1024的行向量。

Parameters:
    filename - 文件名
Returns:
    returnVect - 返回的二进制图像的1*1024行向量
Modify:
    2022-03-29
"""
def img2Vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        #print(lineStr)
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

"""
函数说明：利用scikit-learn实现手写数字分类测试

Parameters:
    无
Returns:
    无
Modify:
    2022-03-29
"""
def handwritingClassTest():
     # 测试集的Labels
     hwLabels = []
     # 返回trainingDigits目录下的文件列表
     trainingFileList = listdir('trainingDigits')
     # 返回文件夹中的文件个数
     m = len(trainingFileList)
     # 初始化一个元素都为零的训练矩阵
     trainingMat = np.zeros((m,1024))
     # 从文件名中解析出训练集的类别
     for i in range(m):
         fileNameStr = trainingFileList[i]
         fileStr = fileNameStr.split('.')[0]
         classNumStr = int(fileStr.split('_')[0])
         hwLabels.append(classNumStr)
         trainingMat[i,:] = img2Vector('trainingDigits/%s' % (fileNameStr))


     # 构建kNN分类器
     neigh = kNN(n_neighbors = 3, algorithm = 'auto')
     # 拟合模型, trainingMat为测试矩阵，hwLabels为对应的标签
     neigh.fit(trainingMat, hwLabels)
     # 返回testDigits目录下的文件列表
     testFileList = listdir('testDigits')
     # 错误计数器
     errorCount = 0.0
     # 测试数据的数量
     mTest = len(testFileList)
     # 从文件中解析出测试集的类别并进行分类测试
     for i in range(mTest):
         fileNameStr = testFileList[i]
         fileStr = fileNameStr.split('.')[0]
         classNumStr = int(fileStr.split('_')[0])
         vectorUnderTest = img2Vector('testDigits/%s' % (fileNameStr))
         #classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
         # 获得预测结果
         classifierResult = neigh.predict(vectorUnderTest)
         print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
         if (classifierResult != classNumStr): errorCount += 1.0
     print("\nthe total number of errors is: %d" % errorCount)
     print("\nthe total error rate is: %f" % (errorCount/float(mTest)))   

  
"""
if __name__ == '__main__':
    returnVect = img2Vector("0_0.txt")
    print(returnVect)
"""



