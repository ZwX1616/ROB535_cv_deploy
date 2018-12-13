# train a model to fit xyz of the vehicle

import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import pandas as pd

batch_size = 512
lr=0.01
wd=0.0005

num_epochs=500

# read data
train_data = pd.read_csv('../data/xyz_train.txt',header=None)
valid_data = pd.read_csv('../data/xyz_valid.txt',header=None)

# prep features and normalize
all_features = pd.concat((train_data.iloc[:, 1:-3], valid_data.iloc[:, 1:-3]))
all_labels = pd.concat((train_data.iloc[:, -3:], valid_data.iloc[:, -3:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

# convert to nd for nn
n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
valid_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(all_labels[:n_train].values)
valid_labels = nd.array(all_labels[n_train:].values)

# loss
loss = gloss.L2Loss()

# metric
def rmse(net, train_features, train_labels):
    preds = net(train_features)
    rmse = nd.sqrt(2 * loss(preds, train_labels).mean())
    return rmse.asscalar()

# define mlp network
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(7, activation='relu'),nn.Dense(3))
#    net.load_parameters('./xyz_v2.1.param', ctx=mx.cpu())
    net.initialize(mx.init.Xavier(), ctx=mx.cpu())
    return net

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    # num_epochs, learning_rate, weight_decay, batch_size
    
    train_ls, valid_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(rmse(net, train_features, train_labels))
        valid_ls.append(rmse(net, valid_features, valid_labels))
        print('epoch '+str(epoch)+' done. rmse='+str(rmse(net, train_features, train_labels))+'/'+str(rmse(net, valid_features, valid_labels)),end='\r')
    return train_ls, valid_ls

# ---
net = get_net()
train_ls, valid_ls = train(net, train_features, train_labels, None, None,
                    num_epochs, lr, wd, batch_size)
gb.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
print('train rmse %f' % train_ls[-1])
print('valid rmse %f' % valid_ls[-1])
net.save_parameters('./xyz_v2.2.param')