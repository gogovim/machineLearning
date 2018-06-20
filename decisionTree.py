import numpy as np
import pandas as pd

class decisionTree:
    def __init__(self,method='ID3'):
        self.T={}
        self.method=method
    def getPro(self,D):
        print("D:",D)
        P=pd.value_counts(D).values/len(D)
        return P
    def infoEntropy(self,D):
        P=self.getPro(D)
        return np.sum(-P*np.log2(P))
    def infoEntropyA(self,Dj,Pj):
        return np.sum(list(map(lambda x,p:p*self.infoEntropy(x),Dj,Pj)))
    def gini(self,D):
        P=self.getPro(D)
        return 1-np.sum(P*P)
    def giniA(self,Dj,Pj):
        return np.sum(list(map(lambda x,p:p*self.gini(x),Dj,Pj)))
    def spiltXOnx(self,X,by):
        print('begin spilt:',X,by)
        uniqueby=np.unique(by)
        print('spilt',list(map(lambda a:X[by==a],uniqueby)))
        return (uniqueby,list(map(lambda a:X[by==a],uniqueby)))
    def gain(self,x,Y,cal1,cal2):
        print('begin gain')
        m=len(Y)
        Dj=self.spiltXOnx(X=Y,by=x)[1]
        Pj=np.array(list(map(lambda a:len(a)/m,Dj)))
        print(Dj)
        print(Pj)
        print(cal1(Dj,Pj))
        return cal2(Y)-cal1(Dj,Pj)
    def ID3(self,X,Y):
        """
        Classification Tree
        Infomation Entropy
        """
        
        print('ID3:\n','X:',X,'Y',Y)
        T={'son':[],'key':None,'value':None,'to':{}}
        if np.unique(Y).size==1:
            T['value']=Y[0]
            return T
        choose=list(map(lambda x:self.gain(x,Y,self.infoEntropyA,self.infoEntropy),np.transpose(X)))
        T['key']=np.argmax(choose)
        print('key:',T['key'])
        print('test X:',X)
        SX=self.spiltXOnx(X,X[:,T['key']])[1]
        SY=self.spiltXOnx(Y,X[:,T['key']])[1]
        keys=self.spiltXOnx(Y,X[:,T['key']])[0]
        T['to']=dict(zip(keys,range(len(keys))))
        T['son']=list(map(lambda x,y:self.ID3(x,y),SX,SY))
        return T
    def findx(self,x,T):
        print(x,T)
        if T['key']==None:
            return T['value']
        return self.findx(x,T['son'][T['to'][x[T['key']]]])
    def findX(self,X,T):
        return np.array(list(map(lambda x:self.findx(x,T),X)))
    def train(self,X,Y):
        if self.method=='ID3':
            self.T=self.ID3(X,Y)
    def predict(self,X):
        if self.method=='ID3':
            Y=self.findX(X,self.T)
        print(X,Y)
        return Y


if __name__=="__main__":
    dt=decisionTree(method='ID3')
    X=np.array([[0,0],[1,1],[0,1],[1,0]])
    Y=np.array([0,0,1,1])
    dt.train(X,Y)
    dt.predict([[0,0],[1,1],[0,1],[1,0]])
    