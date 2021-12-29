import torch
import matplotlib.pyplot as plt
import random


def synthetic_data(w, b, num_examples):
    '''生成数据集'''
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.5, y.shape)
    return X, y.reshape((-1,1))

def line_regression(X, w, b):
    '''定义线性回归模型'''
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y, batch_size):
    '''定义损失函数'''
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / (2 * batch_size)

def sgd_optimizer(params, lr):
    '''定义优化函数'''
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()

def data_iter(X, y, batch_size):
    '''生成数据迭代器'''
    num_examples = len(X)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield X[batch_indices], y[batch_indices]


'''生成训练数据'''
true_w = torch.tensor([3.5])
true_b = 4
train_data, train_labels = synthetic_data(true_w, true_b, 1000)
'''显示训练数据'''
plt.figure(1)
plt.plot(train_data.numpy(), train_labels.numpy(), 'bo', label='train data')
without_noise = torch.matmul(train_data, true_w) + true_b
plt.plot(train_data, without_noise, 'r-', label='without noise')

'''初始化模型参数'''
w = torch.normal(0, 0.01, size=(1, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
'''初始化超参数'''
lr = 0.02
num_epochs = 50
batch_size = 50 # =1, 随机梯度下降；=len(train_data), 经典的梯度下降; =50（自定义）, 批量随机梯度下降
loss = squared_loss
sgd = sgd_optimizer
net = line_regression
'''训练模型'''
plt.figure(2)
plt.ion()
train_loss = torch.zeros(num_epochs)
for epoch in range(num_epochs):
    for X, y in data_iter(train_data, train_labels, batch_size):
        l = loss(net(X, w, b), y, batch_size)
        l.sum().backward()
        sgd([w, b], lr)
    with torch.no_grad():
        train_l = loss(net(train_data, w, b), train_labels, len(train_data))
        print('epoch', epoch, ' loss ', float(train_l.sum()))
        train_loss[epoch] = train_l.sum()
        plt.plot(epoch, train_l.sum(), 'ro')
        plt.show()
        plt.pause(0.1)
plt.plot(range(num_epochs), train_loss,'r-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('batch size=' + str(batch_size) + ', lr=' + str(lr) + ', epochs=' + str(num_epochs))
plt.show()
plt.pause(9999999999999)
plt.close()
