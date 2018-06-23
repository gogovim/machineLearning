import numpy as np
import pandas as pd

from basic import *


class decisionTree:
    def __init__(self,method='ID3'):
        self.T={}
        self.method=method
        self.eps=1e-7

    def spiltXOnx(self,X,by):
        uniqueby=np.unique(by)
        return (uniqueby,list(map(lambda a:X[by==a],uniqueby)))

    def gain(self,x,Y):
        if self.method=='infoGain':
            return infoGain(x,Y)
        if self.method=='infoRatio':
            return infoRatio(x,Y)
            
    def build(self,X,Y,A):
        T={'son':None,'key':None,'value':None,'to':None}
        #print(X,Y,A)
        numY=np.unique(Y)
        if numY.size==0:
            return None
        if numY.size==1:
            T['value']=Y[0]
            return T
        T['value']=pd.value_counts(Y).index[0]
        if A==set():
            return T
        feature=list(A)
        featureGain=list(map(lambda f:self.gain(X[:,f],Y),feature))
        '''
        if np.max(featureGain)<self.eps:
            return T
        '''
        featureIndex=feature[np.argmax(featureGain)]
        #print('feature Index',featureIndex)
        T['key']=featureIndex
        #print(T['key'],featureIndex)
        SX=self.spiltXOnx(X,X[:,featureIndex])[1]
        keyValues,SY=self.spiltXOnx(Y,X[:,featureIndex])
        T['to']=dict(zip(keyValues,range(len(keyValues))))
        A.remove(featureIndex)
        T['son']=list(map(lambda x,y:self.build(x,y,A.copy()),SX,SY))
        return T
    def findx(self,x,T):
        if T['key'] is None:
            return T['value']
        featureIndex=T['key']
        return self.findx(x,T['son'][T['to'][x[featureIndex]]])
    def train(self,X,Y):
        m=X.shape[1]
        A=set(range(m))
        self.T=self.build(X,Y,A)
    def predict(self,X):
        Y=np.array(list(map(lambda x:self.findx(x,self.T),X)))
        print(list(zip(X,Y)))
        return Y

if __name__=="__main__":
    dt=decisionTree(method='infoRatio')

    X=np.array([[0,0],[1,1],[0,1],[1,0]])
    Y=np.array([0,0,1,1])
    dt.train(X,Y)

    dt.predict([[0,0],[1,1],[0,1],[1,0]])
