import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import tensorflow as tf 

class ReLU:
    def __init__(self):
        self.mask = None 
        
    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0 
        return out 
    
    def backward(self, dout):
        dout[self.mask] = 0 
        dx = dout 
        return dx 

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x/sum_exp_x
    
    return y

class Sigmoid:
    def __init__(self):
        self.params = []
        
    def foward(self, x):
        return 1/(1+np.exp(-x))

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # loss
        self.y = None # output of softmax function
        self.t = None # One-hot vector answer label
        
    def forward(self, x, t):
        self.t = t 
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss 
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size 
        
        return dx 

class Affine:
    def __init__(self, w, b):
        self.params = [w,b]
        self.x = None 
        self.dw = None 
        self.db = None 
        
    def forward(self, x):
        w,b = self.params
        self.x = x 
        out = np.matmul(x, w) + b 
        return out 
    
    def backward(self, dout):
        w,b = self.params
        x = self.x 
        dx = np.dot(dout, w.T) # 여기서 대문자 'T'는 전치행렬을 뜻함, 혹은 np.transpose(w) 또는 np.swapaxes(w,0,1)로 정의 가능 
        self.dw = np.dot(x.T, dout)
        self.db = np.sum(dout, axis=0)



class TwoLayerNet:
    
    def __init__(self,input_size, hidden_size, output_size):
        # reset input, output and hidden size 
        I,H,O = input_size, hidden_size, output_size
        
        # reset weight and bias
        W1 = np.random.randn(I,H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H,0)
        b2 = np.random.randn(0)
        
        # create layers
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        
        self.params = []
        for layer in self.layers:
            self.params += layer.params 
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x 
    
class MatMul:
    def __init__(self,W):
        self.params=[W]
        self.grads = [np.zeros_like(W)]
        self.x = None 
    
    def forward(self,x):
        W, = self.params 
        out = np.matmul(x, W)
        self.x = x 
        return out 

    def backward(self,dout):
        w,b = self.params
        x = self.x 
        dx = np.matmul(dout, w.T) # 여기서 대문자 'T'는 전치행렬을 뜻함, 혹은 np.transpose(w) 또는 np.swapaxes(w,0,1)로 정의 가능 
        dw = np.dot(x.T, dout)
        self.grads[0][...] = dw 
        return dx 