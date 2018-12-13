# train a model to do binary image classification

import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from mxnet.gluon.model_zoo import vision
import gluoncv
import numpy as np
import datetime

# ctx=mx.cpu()
ctx=mx.gpu(0)
batch_size = 20
num_epochs = 20

lr=0.0005
wd=0.001

lr_period=5
lr_decay=0.1

# data augmentation (no use)
train_trans = gdata.vision.transforms.Compose([
    # gdata.vision.transforms.RandomResizedCrop((768,432),(0.1,1.0),(1.8,1.6)),
    # gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize(0, 1)])
valid_trans = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize(0, 1)])

train_rec = '../data/train_v3.rec'
valid_rec = '../data/valid_v3.rec'

# loss
# loss = gloss.SoftmaxCrossEntropyLoss()
loss = gluoncv.loss.FocalLoss(num_class=2)

# define bc network
def get_net(ctx=mx.cpu()):
    pre_net = vision.inception_v3(pretrained=True)
    net = vision.inception_v3(classes=2)
    # net.load_parameters('./params_ffn/dbc_15.param', ctx=ctx)
    net.features = pre_net.features
    net.output.initialize(mx.init.Xavier())
    net.collect_params().reset_ctx(ctx)
    net.output.collect_params().setattr('lr_mult', 10)
    # net.initialize(mx.init.Xavier(), ctx=ctx) #
    return net

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period,
          lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_l, train_acc = 0.0, 0.0

        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
            print ('e='+str(epoch)+' lr='+str(trainer.learning_rate * lr_decay))

        iii=0
        for X, y in train_data:
            iii+=1
            print('... {:.1f}% ...'.format(iii*100/len(train_data)),end='\r')
            # X=nd.swapaxes(X,1,3)
            # X=nd.swapaxes(X,2,3)
            y = y.as_in_context(ctx)
#            import pdb; pdb.set_trace() ###
            with autograd.record():
                y_hat = net(X.astype('float32').as_in_context(ctx))
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)
            train_l += l.mean().asscalar()
            train_acc += gb.accuracy(y_hat, y)

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_s = "time %02d:%02d:%02d" % (h, m, s)

        if valid_data is not None:
            valid_acc=0
            iii=0
            for X, y in valid_data:
                iii+=1
                print('... {:.1f}% ...'.format(iii*100/len(valid_data)),end='\r')
                # X=nd.swapaxes(X,1,3)
                # X=nd.swapaxes(X,2,3)
                y = y.as_in_context(ctx)
                val_y_hat = net(X.astype('float32').as_in_context(ctx))
                valid_acc += gb.accuracy(val_y_hat, y)
            epoch_s = ("epoch %d, loss %f, train acc %f, valid acc %f, "
                       % (epoch + 1, train_l / len(train_data),
                          train_acc / len(train_data), valid_acc / len(valid_data)))
        else:
            epoch_s = ("epoch %d, loss %f, train acc %f, " %
                       (epoch + 1, train_l / len(train_data),
                        train_acc / len(train_data)))

        prev_time = cur_time
        print(epoch_s + time_s) # + ', lr ' + str(trainer.learning_rate))
        net.save_parameters('./params_ffn/dbc_'+str(epoch+16)+'.param')
        print('model saved')

# ---
train_dataset=gdata.vision.datasets.ImageRecordDataset(train_rec)
train_dataset=train_dataset.transform_first(train_trans)
valid_dataset=gdata.vision.datasets.ImageRecordDataset(valid_rec)
valid_dataset=valid_dataset.transform_first(valid_trans)

train_data = gdata.DataLoader(train_dataset, batch_size, shuffle=True, last_batch='keep')
valid_data=gdata.DataLoader(valid_dataset, batch_size, shuffle=False, last_batch='keep')

net = get_net(ctx)
net.hybridize()

train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period,
      lr_decay)
