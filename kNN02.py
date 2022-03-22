# -*- coding: UTF-8 -*-

from cProfile import label
from tkinter.font import Font
import numpy as np 
import operator
import matplotlib
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

"""
函数说明：创建数据集

Parameters:
    无
Returns:
    groups - 数据集
    labels - 分类标签
Modify:
    2022-03-13
"""
def createDataSet():
    # 四组二维特征
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    # 四组特征的标签
    labels = ['爱情片','爱情片','动作片','动作片']
    return group, labels


"""
函数说明: kNN算法, 分类器

Parameters:
    inX - 用于分类的数据（测试集）
    dataSet - 用于训练的数据（训练集）
    labels - 分类标签
    k - kNN算法参数, 选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
Modify:
    2022-03-13
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
函数说明: 打开并解析文件，对数据进行分类：1代表不喜欢，2代表魅力一般，3代表极具魅力

Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Labe向量
Modify:
    2022-03-21
"""
def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # 读取文件所有内容
    arrayOLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOLines)
    # 返回的NumPy矩阵，解析完成的数据：numberOfLines行，3列
    returnMat = np.zeros((numberOfLines,3))
    # 返回的分类标签向量
    classLabelVector = []
    # 行的索引值
    index = 0
    for line in arrayOLines:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        # classLabelVector.append(int(listFromLine[-1]))
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector        

"""
函数说明：可视化数据

Parameters:
    datingDataMat - 特征矩阵
    datingLabels - 分类标签
Returns:
    无
Modify:
    2022-03-21
"""
def showdata(datingDataMat, datingLabels):
    # 设置汉字格式
    font = FontProperties(fname='C:/windows/fonts/simsun.ttc', size=14)
    # 将fig画布分隔成1行1列，不共享x轴和y轴，fig画布的大小为(13,8)
    # 当nrows=2, ncols=2时，代表fig画布被分成四个区域，axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13,8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')

    # 画出散点图，以datingDataMat矩阵的第一(飞行常客里程)、第二列(玩游戏)数据画散点数据，散点大小为15，透明度为0.5   
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors, s=15, alpha=0.5)
    # 设置标题，x轴label，y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图，以datingDataMat矩阵的第一(飞行常客里程)、第三列(冰激凌)数据画散点数据，散点大小为15，透明度为0.5   
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors, s=15, alpha=0.5)
    # 设置标题，x轴label，y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图，以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据，散点大小为15，透明度为0.5   
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors, s=15, alpha=0.5)
    # 设置标题，x轴label，y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')

    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])

    # 显示图片  
    plt.show()

"""
函数说明：归一化函数

Parameters:
    dataSet - 数据集
Returns:
    normDataSet - 归一化后的数据集
    ranges - 数据范围
    minVals - 最小值
Modify:
    2022-03-21
"""
def autoNorm(dataSet):
    # 返回数据集的最小值
    # min(0)返回每一列的最小值；min(1)返回每一行的最小值
    minVals = dataSet.min(0) # 返回值为行向量，列数与数据集的列数相同
    # 返回数据集的最大值
    # max(0)返回每一列的最大值；max(1)返回每一行的最大值
    maxVals = dataSet.max(0) # 返回值为行向量，列数与数据集的列数相同
    ranges = maxVals - minVals
    # 定义一个和数据集维度一样，取值均为0的空数据集
    mn = dataSet.shape
    normDataSet = np.zeros(mn)
    # 获取数据集的行数
    m = dataSet.shape[0]
    # 先将minVals沿着X轴方向复制1倍，沿着Y轴方向复制m倍
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normMat = normDataSet/np.tile(ranges,(m,1))
    return normMat, ranges, minVals

"""
函数说明：main函数

Parameters:
    无
Returs:
    无
Modify:
    2022-03-21
"""
if __name__ == '__main__':
  # 打开的文件名
  filename = "datingTestSet.txt"
  # 打开并处理数据
  datingDataMat, datingLabels = file2matrix(filename)
  normMat, ranges, minVals = autoNorm(datingDataMat)
  showdata(datingDataMat, datingLabels)
  print('normMat = ', normMat)
  print('ranges = ', ranges)
  print('minVals = ', minVals)

    