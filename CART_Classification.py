import numpy as np
import pandas as pd

from basic import *


class CART_Classification:
    def __init__(self):
        self.T={}
        self.featureType=[]
    def getSpiltPoint(self,f,t):
        f=np.unique(f)
        return np.linspace(f.min(),f.max(),np.sqrt(len(f)),False) if t else f
    def getSplitArr(self,f,v,t):
        return f<=v if t else f==v
    def getSplitInterval(self,X,by):
        #print(X,by)
        return (X[by],X[by^True])
    def getGini(self,D):
        d0=len(D[0])
        d1=len(D[1])
        d=d0+d1
        return giniA(D,[d0/d,d1/d])
    def gain(self,f,Y,t):
        #print('gain:',f,Y)
        spiltPoint=self.getSpiltPoint(f,t)
        splitArr=list(map(lambda cp:self.getSplitArr(f,cp,t),spiltPoint))
        splitInterval=list(map(lambda arr:self.getSplitInterval(Y,arr),splitArr))
        splitGini=list(map(lambda D:self.getGini(D),splitInterval))
        sp=np.argmin(splitGini)
        return spiltPoint[sp],splitGini[sp]
 
    def build(self,X,Y):
        #print('build ',X,Y)
        T={'left':None,'right':None,'key':None,'keyValue':None,'value':None}
        numY=np.unique(Y)
        if numY.size==0:
            return None
        if numY.size==1:
            T['value']=Y[0]
            return T
        T['value']=pd.value_counts(Y).index[0]
        splitFeature=np.array(list(map(lambda f,t:self.gain(f,Y,t),np.transpose(X),self.featureType)))
        featureIndex=np.argmin(splitFeature[:,1])
        #print('splitFeature:',splitFeature)
        #print('featureIndex:',featureIndex)
        T['key']=featureIndex
        T['keyValue']=splitFeature[featureIndex,0]

        spiltArr=self.getSplitArr(X[:,featureIndex],T['keyValue'],self.featureType[featureIndex])
        T['left'],T['right']=map(lambda A:self.build(A[0],A[1]),
        [(X[spiltArr],Y[spiltArr]),(X[spiltArr^True],Y[spiltArr^True])])
        return T


    def findx(self,x,T):
        if T['key'] is None:
            return T['value']
        featureIndex=T['key']
        if self.featureType[featureIndex]:
            next=T['left'] if x[featureIndex]<=T['keyValue'] else T['right']
        else:
            next=T['left'] if x[featureIndex]==T['keyValue'] else T['right']
        return T['value'] if next is None else self.findx(x,next)
    def train(self,X,Y):
        self.featureType=getFeatureType(X)
        self.T=self.build(X,Y)
    def predict(self,X):
        Y=np.array(list(map(lambda x:self.findx(x,self.T),X)))
        print(list(zip(X,Y)))
        return Y
if __name__=="__main__":
    dt=CART_Classification()

    X=np.array([[0,0],[1,1],[0,1],[1,0]])
    Y=np.array([0,0,1,1])
    dt.train(X,Y)

    dt.predict([[0,0],[1,1],[0,1],[1,0]])
    dt.predict([[0.5,0.5]])
