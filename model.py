import math
import numpy as np
#函数类
class shenjingwangluo:
    def __init__(self, input_dim, hidden_dim, output_dim, regularization_intensity):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.regularization_intensity = regularization_intensity

        self.w1 = np.random.randn(self.hidden_dim, self.input_dim)
        self.b1 = np.random.randn(self.hidden_dim)
        self.w2 = np.random.randn(self.output_dim, self.hidden_dim)
        self.b2 = np.random.randn(self.output_dim)
    #正向
    def __call__(self, data):
        self.x = data
        self.z1 = np.add(np.dot(self.w1, data.T), self.b1.reshape(-1, 1))
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.add(np.dot(self.w2, self.a1), self.b2.reshape(-1, 1))
        self.a2 = self.softmax(self.z2.T)
        self.y_pred = np.argmax(self.a2, axis=1)
        return self.y_pred
    #反向
    def backward(self, open, epoch):
        terninal = 0.0001
        decay = 200
        lr = terninal + (open - terninal) * math.exp(-epoch/decay)
        self.z2_grad = self.a2.copy()
        n = len(self.y_true)
        self.z2_grad[np.arange(n), self.y_true] -= 1
        self.w2_grad = self.z2_grad.T.reshape(self.output_dim, 1, len(self.y_true)) * self.a1
        self.b2_grad = self.z2_grad.T
        # self.a1_grad = np.dot(self.z2_grad, self.w2)
        self.z1_grad = np.dot(self.z2_grad, self.w2).T * self.a1 * (1-self.a1)
        self.w1_grad = self.z1_grad.reshape(self.hidden_dim, 1, len(self.y_true)) * self.x.T
        self.b1_grad = self.z1_grad
        self.w2_grad = np.sum(self.w2_grad, axis=2) + 2 * self.regularization_intensity * self.w2
        self.w1_grad = np.sum(self.w1_grad, axis=2) + 2 * self.regularization_intensity * self.w1
        self.b2_grad = np.sum(self.b2_grad, axis=1) + 2 * self.regularization_intensity * self.b2
        self.b1_grad = np.sum(self.b1_grad, axis=1) + 2 * self.regularization_intensity * self.b1
        self.w2 -= lr * self.w2_grad
        self.b2 -= lr * self.b2_grad
        self.w1 -= lr * self.w1_grad
        self.b1 -= lr * self.b1_grad
    #损失函数
    def loss(self, y_true):
        self.y_true = y_true
        n = len(y_true)
        loss = -np.sum(np.log(self.a2[np.arange(n), y_true])) / n
        return loss
    #激活函数
    @staticmethod
    def sigmoid(x):
        # return 1 / (1+np.exp(-x))
        if np.all(x >= 0):
            return 1. / (1.+np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))
    @staticmethod
    def softmax(x):
        return np.divide(np.exp(x), np.sum(np.exp(x), axis=1).reshape(-1, 1))
    
    def save(self, path):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
    def load(self, path):
        parameters = np.load(path)
        self.w1 = parameters['w1']
        self.b1 = parameters['b1']
        self.w2 = parameters['w2']
        self.b2 = parameters['b2']
