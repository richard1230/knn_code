
import numpy as np
import operator

def createDataSet():
    matrix = np.array([[1.0,1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])##创建矩阵
    classVector = ['A','A','B','B']                                   ##标明类别
    return matrix,classVector

matrix,classVector = createDataSet();

##算法封装
def classify(inX ,matrix,classvector,k):              ##inX为待检测目标数据,k指的是样本集中最相似的前K个数据
    dataSetSize = matrix.shape[0]                     ##matrix.shape[0]代表的是matrix这个矩阵的行数,matrix.shape[1]代表的是matrix这个矩阵的列数
    diffmat = np.tile(inX,(dataSetSize,1)) - matrix  ##做差
    sqDiffMat= diffmat ** 2                          ## 平方
    sqDistances = sqDiffMat.sum(axis=1)              ##求平方和
    distance = sqDistances ** 0.5                    ##求出每个样本与未知样本的距离
    sortDistIndicies = distance.argsort()            ##根据下标来进行排序;argsort函数是将distance中的元素从小到大排列，提取其对应的index(索引)，然后输出,注意:这里将所有距离都给求出来了(并且是按照从小到大来排的)！！！
    print(sortDistIndicies)
    classCount = {}                                  ##自定义了一个字典,用于保存{A:2,B:1},里面的A，B为种类,2,1为下标(索引)
    for i in range(k):
        voteLabel = classVector[sortDistIndicies[i]] ##
        classCount[voteLabel]=classCount.get(voteLabel,0)+1   ##这里的get方法为字典里面的方法，0表示如果不存在key则返回0；A，B变成了{'A': 2, 'B': 1},这里是一个非常重要的技巧,开发会用到
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)##itemgetter(1)表示是按照{'A': 2, 'B': 1}里面的2,1来排序的，如果为itemgetter(0)表示是按照{'A': 2, 'B': 1}里面的A，B来排序的
    ##sortedClassCount是个数组
    print(sortedClassCount[0][0])



classify([1, 1], matrix, classVector, 3)

##总结
# 这边其实已经把所有距离都算出来了，默认是从小到大排序的，k只不过是取得前k个最短的距离值
# 然后用字典来计数的,按照从大到小的顺序排，显然第一个就是票数最多的





# labelCounts[currentLabel] += 1                                          #这里的这个技巧同Knn,利用字典进行计数
#         # print(labelCounts)
#         # labelCounts[currentLabel]=labelCounts.get(currentLabel,0) +1          #等价于上上面一行

