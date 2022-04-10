from process import data, mini_batch, _accuracy
from model import shenjingwangluo
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import xlwt
class library:
    def __init__(self):
        self.seed = 1
        self.hidden = 256
        self.epoches = 200
        self.batch_size = 64
        self.start = 0.1
        self.intensity = 0.005

def train(hero):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    model_path = f'{curr_path}/output/lr{hero.start}_hd{hero.hidden}_ri{hero.intensity}/'

    np.random.seed(hero.seed)
    train_X, train_y, test_X, test_y = data('./data/mnist.pkl.gz')
    n, input_dim = train_X.shape
    model = shenjingwangluo(input_dim, hero.hidden, 10, hero.intensity)
    loss_train_total = []
    loss_test_total = []
    acc_total = []
    print(f'learning-rate:{hero.start}, hidden-layer:{hero.hidden}, dropout:{hero.intensity}')
    for epoch in range(hero.epoches):
        batch_indices = mini_batch(len(test_X), hero.batch_size)
        batch_num = 0
        loss_epoch = 0
        for batch in batch_indices:
            batch_num += 1
            train_X_batch = train_X[batch]
            model(train_X_batch)
            y_true = train_y[batch]
            loss_batch = float(model.loss(y_true))
            loss_epoch += 1 / len(batch_indices) * (loss_batch - loss_epoch)
            model.backward(hero.start, epoch)
        loss_train_total.append(loss_epoch)
        test_y_predict = model(test_X)
        test_loss = model.loss(test_y)
        loss_test_total.append(test_loss)
        acc = _accuracy(test_y, test_y_predict)
        acc_total.append(acc)
        print(f'epoch:{epoch+1}/{hero.epoches}\t   train_loss:{round(loss_epoch, 2)}\t  test_loss:{round(test_loss, 2)}\t acc:{np.round(acc*100, 2)}%.')
    Path(model_path).mkdir(parents=True, exist_ok=True)
    plot_a(model_path, loss_train_total, loss_test_total)
    plot_b(model_path, acc_total)
    plot_c(model_path, loss_train_total, loss_test_total, acc_total)
    model.save(model_path+'parameters')

def plot_a(path, loss_train, loss_test):
    plt.figure(dpi=150)
    plt.title('Loss Curve')
    plt.plot(loss_train)
    plt.plot(loss_test)
    plt.legend(['train', 'test'])
    plt.savefig(path+'LossCurve.jpg')

def plot_b(path, acc):
    plt.figure(dpi=150)
    plt.title('_Accuracy Curve')
    plt.plot(acc)
    plt.savefig(path+'AccCurve.jpg')

def plot_c(path, train_loss, test_loss, acc):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)

    names = ['train_loss', 'test_loss', 'acc']
    for j in range(len(names)):
        sheet1.write(0, j, names[j])
        for i in range(len(acc)):
            sheet1.write(i+1, j, eval(names[j])[i])
    f.save(path+'metircs.xlsx')

def show(data):
    plt.imshow(data.reshape((20, 20)), cmap='red')

if __name__ == '__main__':
    for i in [0.1, 0.01, 0.001]:
        for hidden in [50, 100, 200]:
            for intensity in [0.1, 0.01, 0.001]:
                hero = library()
                hero.start = i
                hero.hidden = hidden
                hero.intensity = intensity
                train(hero)
