import os
import sys
import csv

# training parameters #

data_shape='512 384' #1536,864
label_width='128'
# tensorboard='True'
batch_per_gpu=8 #cgpu
initial_learning_rate='0.001'
lr_decay='0.316' # lr decay rate
num_lr_decay=8 # number of times that lr decays
lr_step=6 # lr decay period
network='densenet-tiny-oi-cc'
# ------------------ #

str_args=''
str_args+=' --data-shape '+data_shape
str_args+=' --label-width '+label_width
# str_args+=' --tensorboard '+tensorboard
str_args+=' --lr '+initial_learning_rate
str_args+=' --lr-factor '+lr_decay
lr_steps=str(lr_step)
for i in range(num_lr_decay-1):
	lr_steps+=','+str((i+2)*lr_step)
str_args+=' --lr-steps '+lr_steps
end_epoch=str((num_lr_decay+1)*lr_step)
str_args+=' --end-epoch '+end_epoch
# print(lr_steps+' '+end_epoch+' '+str(float(initial_learning_rate)*float((float(lr_decay)**num_lr_decay))))
str_args+=' --network '+network

str_args+=' --gpus 0'

batch_size=str(int(1*batch_per_gpu))
str_args+=' --batch-size '+batch_size

train_path='../data/train.rec'
str_args+=' --train-path '+train_path

valid_path='../data/valid.rec'
str_args+=' --val-path '+valid_path


prefix='../params/gta5_2d'
str_args+=' --prefix '+prefix

str_args+=' --num-class '+str(3) # str(6)
str_args+=' --class-names '+'0,1,2' # '0,1,2,3,4,5'

with open('../params/run.log','a+') as wf:
	wf.write('python train.py '+str_args+'\n')

print('\n'+'python train.py '+str_args+'\n')
print('logged to '+'../params/run.log')
os.system('python train.py '+str_args)
# 

