import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import copy
from para_search import grid

# 定义激活函数及其求导
def relu(x):
    return np.where(x >= 0, x, 0)


def d_relu(x):
    return np.where(x >= 0, 1, 0)


def softmax(x):
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


def d_softmax(x):
    sm = softmax(x)
    return np.diag(sm) - np.outer(sm, sm)



activation = [relu, softmax]
differential = {softmax: d_softmax, relu: d_relu}
distribution = [
    {'b': [0, 0], 'w': [-1, 1]},
    {'b': [0, 0], 'w': [-1, 1]},
]


# 参数初始化
def init_parameters_b(layer):
    dist = distribution[layer]['b']
    return np.random.rand(dimensions[layer + 1]) * (dist[1] - dist[0]) + dist[0]


def init_parameters_w(layer):
    dist = distribution[layer]['w']
    return np.random.rand(dimensions[layer], dimensions[layer + 1]) * (dist[1] - dist[0]) + dist[0]


def init_parameters():
    parameter = []
    for i in range(len(distribution)):
        layer_parameter = {}
        for j in distribution[i].keys():
            if j == 'b':
                layer_parameter['b'] = init_parameters_b(i)
                continue
            if j == 'w':
                layer_parameter['w'] = init_parameters_w(i)
                continue
        parameter.append(layer_parameter)
    return parameter


# 前向传播
def predict(img,parameters):
    hidden_in=np.dot(img,parameters[0]['w'])+parameters[0]['b']
    hidden_out=activation[0](hidden_in)
    l0_in=np.dot(hidden_out,parameters[1]['w'])+parameters[1]['b']
    l0_out=activation[1](l0_in)
    return l0_out


# 损失函数以及L2正则化
onehot=np.identity(10)


def sqr_loss(img,lab,parameters, ll):
    ll = ll
    y_pred=predict(img,parameters)
    y=onehot[lab]
    diff=y-y_pred
    return 0.5*np.dot(diff,diff)+ll*np.sum(np.square(parameters[0]['w']))+ll*np.sum(np.square(parameters[1]['w']))


# BP
def grad_parameters(img, lab, parameters, ll):
    ll = ll
    hidden_in = np.dot(img, parameters[0]['w']) + parameters[0]['b']
    hidden_out = activation[0](hidden_in)
    last_in = np.dot(hidden_out, parameters[1]['w']) + parameters[1]['b']
    last_out = activation[1](last_in)

    diff = onehot[lab] - last_out
    act1 = np.dot(differential[activation[1]](last_in), diff)

    grad_b1 = -2 * act1
    grad_w1 = -2 * (np.outer(hidden_out, act1) + ll * parameters[1]['w'])
    grad_b0 = -2 * differential[activation[0]](hidden_in) * np.dot(parameters[1]['w'], act1)
    # print(grad_b0.shape)
    grad_w0 = -2 * (np.outer(img, (differential[activation[0]](hidden_in) * np.dot(parameters[1]['w'], act1))) + ll *
                    parameters[0]['w'])

    return {'w1': grad_w1, 'b1': grad_b1, 'w0': grad_w0, 'b0': grad_b0}

# 读入数据
train_num=50000
valid_num=10000
test_num=10000

dataset_path=Path('D:/hw1')
train_img_path=dataset_path/'train-images.idx3-ubyte'
train_lab_path=dataset_path/'train-labels.idx1-ubyte'
test_img_path=dataset_path/'t10k-images.idx3-ubyte'
test_lab_path=dataset_path/'t10k-labels.idx1-ubyte'

with open(train_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    tmp_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28) / 255
    train_img = tmp_img[:train_num]
    valid_img = tmp_img[train_num:]

with open(test_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    test_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28) / 255

with open(train_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    tmp_lab = np.fromfile(f, dtype=np.uint8)
    train_lab = tmp_lab[:train_num]
    valid_lab = tmp_lab[train_num:]

with open(test_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    test_lab = np.fromfile(f, dtype=np.uint8)

# 可视化
def show_train(index):
    plt.imshow(train_img[index].reshape(28,28),cmap='gray')
    print('label : {}'.format(train_lab[index]))


def show_valid(index):
    plt.imshow(valid_img[index].reshape(28,28),cmap='gray')
    print('label : {}'.format(valid_lab[index]))


def show_test(index):
    plt.imshow(test_img[index].reshape(28,28),cmap='gray')
    print('label : {}'.format(test_lab[index]))


# 指标
def valid_loss(parameters,ll):
    loss_accu=0
    for img_i in range(valid_num):
        loss_accu+=sqr_loss(valid_img[img_i],valid_lab[img_i],parameters,ll)
    return loss_accu/(valid_num/10000)
def valid_accuracy(parameters):
    correct=[predict(valid_img[img_i],parameters).argmax()==valid_lab[img_i] for img_i in range(valid_num)]
    return correct.count(True)/len(correct)
def train_loss(parameters,ll):
    loss_accu=0
    for img_i in range(train_num):
        loss_accu+=sqr_loss(train_img[img_i],train_lab[img_i],parameters,ll)
    return loss_accu/(train_num/10000)
def train_accuracy(parameters):
    correct=[predict(train_img[img_i],parameters).argmax()==train_lab[img_i] for img_i in range(train_num)]
    return correct.count(True)/len(correct)
def test_accuracy(parameters):
    correct=[predict(test_img[img_i],parameters).argmax()==test_lab[img_i] for img_i in range(test_num)]
    return correct.count(True)/len(correct)

# 训练函数
batch_size=100
def train_batch(current_batch,parameters,ll):
    grad_accu=grad_parameters(train_img[current_batch*batch_size+0],train_lab[current_batch*batch_size+0],parameters,ll)
    for img_i in range(1,batch_size):
        grad_tmp=grad_parameters(train_img[current_batch*batch_size+img_i],train_lab[current_batch*batch_size+img_i],parameters,ll)
        for key in grad_accu.keys():
            grad_accu[key]+=grad_tmp[key]
    for key in grad_accu.keys():
        grad_accu[key]/=batch_size
    return grad_accu

# 更新参数
def combine_parameters(parameters,grad,learn_rate):
    parameter_tmp=copy.deepcopy(parameters)
    parameter_tmp[0]['b']-=learn_rate*grad['b0']
    parameter_tmp[0]['w']-=learn_rate*grad['w0']
    parameter_tmp[1]['b']-=learn_rate*grad['b1']
    parameter_tmp[1]['w']-=learn_rate*grad['w1']
    return parameter_tmp


if __name__ == '__main__':
    grid_accu = [-1]
    for grid_i in range(2):
        print(f'载入第{grid_i+1}组超参')
        ll = grid['L2'][grid_i]
        hidden_dim = grid['hidden_dims'][grid_i]
        lr = grid['lr'][grid_i]
        dimensions = [784, hidden_dim, 10]
        # 初始化参数
        parameters = init_parameters()
        current_epoch = 0
        train_loss_list = []
        valid_loss_list = []
        train_accu_list = []
        valid_accu_list = []

        # 训练
        learn_rate = lr
        epoch_num = 15
        for epoch_ in range(epoch_num):
            for i in range(train_num // batch_size):
                grad_tmp = train_batch(i, parameters,ll)
                parameters = combine_parameters(parameters, grad_tmp, learn_rate)
            current_epoch += 1
            tl = train_loss(parameters,ll)
            ta = train_accuracy(parameters)
            vl = valid_loss(parameters,ll)
            va = valid_accuracy(parameters)
            train_loss_list.append(tl)
            train_accu_list.append(ta)
            valid_loss_list.append(vl)
            valid_accu_list.append(va)
            print(f'当前epoch{current_epoch}/{epoch_num}, 训练损失：{tl}, 验证损失：{vl}, 训练准确率：{ta}, 验证准确率：{va}')

        # 验证集准确率
        accu = valid_accuracy(parameters)


        # 损失准确率曲线
        fig = plt.figure(figsize=(12, 4), dpi=100)
        ax1 = fig.add_subplot(1, 2, 1)
        plt.plot(valid_accu_list, c='r', label='val acc')
        plt.plot(train_accu_list, c='b', label='train acc')
        plt.legend()

        ax2 = fig.add_subplot(1,2,2)
        plt.plot(valid_loss_list, c='r', label='val loss')
        plt.plot(train_loss_list, c='b', label='train loss')
        plt.legend()
        plt.savefig(f'./grid_{grid_i})_acc_loss')



        import pickle

        # 在多组超参中选取验证结果最好的模型保存
        if accu > max(grid_accu):
            model_prameters_name = './best_model.pkl'
            f = open(model_prameters_name, 'wb')
            pickle.dump(parameters, f)
            f.close()
        grid_accu.append(accu)
    print(f'各组超参的验证集准确率为{grid_accu}')