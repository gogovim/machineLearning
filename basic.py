import numpy as np
import pandas as pd

#判断数据集中各属性是否是连续值,True为连续值,False为离散值
def getFeatureType(X):
    m=np.sqrt(len(X))
    return list(map(lambda f:np.unique(f).size>m,np.transpose(X)))
#计算序列中各元素所占的比例
def getPro(D):
    return pd.value_counts(D).values/len(D)
#计算序列的熵
def infoEntropy(D):
    P=getPro(D)
    return np.sum(-P*np.log2(P))
#根据标记by,将X分成几个部分,
def spiltXOnx(X,by):
    uniqueby=np.unique(by)
    return (uniqueby,list(map(lambda a:X[by==a],uniqueby)))
#计算序列Y用x分类后熵的增益
def infoGain(x,Y):
    m=len(Y)
    Dj=spiltXOnx(X=Y,by=x)[1]
    Pj=np.array(list(map(lambda a:len(a)/m,Dj)))
    return infoEntropy(Y)-np.sum(list(map(lambda x,p:p*infoEntropy(x),Dj,Pj)))
#计算序列Y用x分类后熵的增益率
def infoRatio(x,Y):
    return infoGain(x,Y)/infoEntropy(x)
#计算gini指数
def gini(D):
    P=getPro(D)
    return 1-np.sum(P*P)
#计算在特征A条件下,集合D的gini指数
def giniA(D,P):
    return np.sum(list(map(lambda x,p:p*gini(x),D,P)))

