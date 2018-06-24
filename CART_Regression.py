import numpy as np
import pandas as pd

from basic import getFeatureType

class CART_Regression:
    def __init__(self):
        self.T={}
        self.featureType=[]
        self.eps=1e-7
    def getSpiltPoint(self,f,t):
        f=np.unique(f)
        return np.linspace(f.min(),f.max(),np.sqrt(len(f)),False) if t else f
    def getSplitArr(self,f,v,t):
        return f<=v if t else f==v
    def getSplitInterval(self,X,by):
        #print(X,by)
        return (X[by],X[by^True])
    def getLeastSquares(self,D):
        c0=np.mean(D[0])
        c1=np.mean(D[1])
        return np.sum(np.square(D[0]-c0))+np.sum(np.square(D[1]-c1))
    def gain(self,f,Y,t):
        #print('gain:',f,Y)
        spiltPoint=self.getSpiltPoint(f,t)
        splitArr=list(map(lambda cp:self.getSplitArr(f,cp,t),spiltPoint))
        splitInterval=list(map(lambda arr:self.getSplitInterval(Y,arr),splitArr))
        splitLeastSquare=list(map(lambda D:self.getLeastSquares(D),splitInterval))
        sp=np.argmin(splitLeastSquare)
        print(spiltPoint[sp],splitLeastSquare[sp])
        return spiltPoint[sp],splitLeastSquare[sp]
 
    def build(self,X,Y):
        #print('build ',X,Y)
        T={'left':None,'right':None,'key':None,'keyValue':None,'value':None}
        numY=np.unique(Y)
        if numY.size==0:
            return None
        T['value']=np.mean(Y)
        if np.sum(np.square(Y-np.mean(Y)))<self.eps:
            return T
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

    def getG(self,T,X,Y):
        G={'g':None,'left':None,'right':None}
        if T['key']==None:
            G['g']=np.Inf
            return G['g'],np.sum(np.square(Y-T['value'])),len(Y),G
        featureIndex=T['key']
        spiltArr=self.getSplitArr(X[:,featureIndex],T['keyValue'],self.featureType[featureIndex])
        lG,lC,lNum,G['left']=self.getG(T['left'],X[spiltArr],Y[spiltArr])
        rG,rC,rNum,G['right']=self.getG(T['right'],X[spiltArr^True],Y[spiltArr^True])
        G['g']=(np.sum(np.square(Y-T['value']))-lC-rC)/(lNum+rNum)
        return min(lG,rG,G['g']),lC+rC,lNum+rNum,G
    def cutByalpha(self,T,G,alpha):
        if T['key']==None:
            return T
        if G['g']<=alpha:
            T['left']=T['right']=T['key']=T['keyValue']=None
            return T
        T['left']=self.cutByalpha(T['left'],G['left'],alpha)
        T['right']=self.cutByalpha(T['right'],G['right'],alpha)
        return T
    def copyT(self,T):
        if T['key']==None:
            return T.copy()
        rT=T.copy()
        rT['left']=self.copyT(T['left'])
        rT['right']=self.copyT(T['right'])
        return rT
    def calTError(self,X,Y,T):
        self.T=T
        preY=self.predict(X)
        return np.sum(np.square(preY-Y))
    def pruningTree(self,X,Y,valX,valY):
        teT=self.copyT(self.T)
        TList=[]
        
        while teT['key']!=None:
            minG,c,num,G=self.getG(teT,X,Y)
            teT=self.cutByalpha(teT,G,minG)
            TList.append(self.copyT(teT))
            print('g:',minG,'G:',G)
        TlistError=list(map(lambda T:self.calTError(valX,valY,T),TList))
        self.T=TList[np.argmin(TlistError)]
    def findx(self,x,T):
        if T['key'] is None:
            return T['value']
        featureIndex=T['key']
        if self.featureType[featureIndex]:
            next=T['left'] if x[featureIndex]<=T['keyValue'] else T['right']
        else:
            next=T['left'] if x[featureIndex]==T['keyValue'] else T['right']
        return T['value'] if next is None else self.findx(x,next)
    def train(self,X,Y,valX=None,valY=None):
        self.featureType=getFeatureType(X)
        self.T=self.build(X,Y)
        if valX is not None and valY is not None:
            self.pruningTree(X,Y,valX,valY)
    def predict(self,X):
        Y=np.array(list(map(lambda x:self.findx(x,self.T),X)))
        print(list(zip(X,Y)))
        return Y
if __name__=="__main__":
    dt=CART_Regression()

    X=np.array([[0,0],[1,1],[0,1],[1,0],[1,1]])
    Y=np.array([0,0,1,1,0])
    print(X[:2],Y[:2])
    dt.train(X,Y,X[:2],Y[:2])

    dt.predict([[0,0],[1,1],[0,1],[1,0]])
    dt.predict([[0.5,0.5]])
