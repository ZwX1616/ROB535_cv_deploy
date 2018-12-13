# train a model to do binary image classification

# import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from mxnet.gluon.model_zoo import vision
import numpy as np
import datetime
import csv

# ctx=mx.cpu()
ctx=mx.gpu(0)

batch_size = 20
# lr=0.01
# wd=0.0005

epoch=32
# num_epochs = 2
# lr_period=5
# lr_decay=0.1

test_rec = '../data/test_v3.rec'
test_trans = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize(0, 1)])

# define bc network
def get_net(ctx=mx.cpu()):
    net = vision.Inception3(classes=2)
    return net

# ---
test_dataset=gdata.vision.datasets.ImageRecordDataset(test_rec)
test_dataset=test_dataset.transform_first(test_trans)

test_data = gdata.DataLoader(test_dataset, batch_size, shuffle=False, last_batch='keep')

net = get_net(ctx)
net.load_parameters('./params_ffn/dbc_'+str(epoch)+'.param', ctx=ctx)
net.hybridize()

preds = []
iii=0
for X, _ in test_data:
    iii+=1
    print(iii/2631*batch_size*100,end='\r')
    # X=nd.swapaxes(X,1,3)
    # X=nd.swapaxes(X,2,3)
    y_hat = net(X.astype('float32').as_in_context(ctx))
    preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())

# write result
with open('../data/test_v3.lst') as cfile: # read to get filenames, ordered
    reader = csv.reader(cfile)
    readeritem=[]
    readeritem.extend([row for row in reader])
    
wf=open('../output/output_t1_v3_u.txt','w+',newline='')
writer=csv.writer(wf)
writer.writerow(['guid/image','label'])

for i,rows in enumerate(readeritem):
	s=rows[0].strip().split("\t")
	s=s[2]
	file_id=s[-51:-15]
	file_idx=s[-14:-10]
	writer.writerow([file_id+'/'+file_idx,str(preds[i]+1)])
