# get feature from labels and predict xyz

import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import pandas as pd
import csv

# read data
train_data = pd.read_csv('../data/xyz_train.txt',header=None)
valid_data = pd.read_csv('../data/xyz_valid.txt',header=None)
all_features = pd.concat((train_data.iloc[:, 0:-3], valid_data.iloc[:, 0:-3]))
fmean=np.array(all_features.mean())
fstd=np.array(all_features.std())
# prep features and normalize

test_data = pd.read_csv('../output/output_t2.txt',header=0)
features=[]
for i in range(test_data.shape[0]):
#for i in range(1000):
    guid=test_data[:].values[i][0]
    cls=float(test_data[:].values[i][2])
    xmin=float(test_data[:].values[i][3])
    ymin=float(test_data[:].values[i][4])
    xmax=float(test_data[:].values[i][5])
    ymax=float(test_data[:].values[i][6])
    area=((xmax-xmin)*(ymax-ymin))
    proj = np.fromfile('../data/test/'+guid+'_proj.bin', dtype=np.float32)
    proj.resize([3, 4])
    p=proj.flatten()
    features.append([cls,xmin,ymin,xmax,ymax,area,p[0],p[1],p[2],p[5],p[6]])

features=np.array(features)
features= (features - fmean) / fstd
test_features = nd.array(features)


#Out[37]: -1.8518614481394198
#

#Out[38]: -2.9768518344027775
#

#Out[39]: 36.597185685776125
    
# define mlp network
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(7, activation='relu'),nn.Dense(3))
    net.load_parameters('./xyz_v2.2.param', ctx=mx.cpu())
    return net

# ---
net = get_net()
preds = net(test_features)

wf=open('./output/t2output.txt','w+',newline='')
writer=csv.writer(wf)
writer.writerow(['guid/image/axis','value']) ## for task2
gc=0
for i in range(preds.shape[0]):
    print(i,end='\r')
    guid=test_data[:].values[i][0]
    if test_data[:].values[i][6]==-1:
        x=-1.851861
        y=-2.976852
        z=36.597186
        gc+=1
    else:
        x=preds.asnumpy()[i,0]
        y=preds.asnumpy()[i,1]
        z=preds.asnumpy()[i,2]
    writer.writerow([guid+'/x',x])
    writer.writerow([guid+'/y',y])
    writer.writerow([guid+'/z',z])
print(i,gc)