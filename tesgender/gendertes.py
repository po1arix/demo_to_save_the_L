import torch
from torch.autograd import Variable
from train_sex import SexNet


use_cuda = False
model = SexNet()
model.load_state_dict(torch.load('output/params_99.pth'))  #params_xx.pth任意选择一个
model.eval()

f,f2 = open('whatisit.txt', 'r'),open('result.txt','w') 

for line in f:
    line = line.strip('\n')
    data = line.split()
    olddata = data
    data_tensor = torch.FloatTensor([float(data[0])/2.0,float(data[1])/80.0])
    data_tensor = data_tensor.unsqueeze(0)
    prediction = model(Variable(data_tensor))
    print(prediction)
    pred = torch.max(prediction, 1)[1]\
    
    predy = pred.data.numpy()
    if predy == 0:
        print(olddata[0],olddata[1],"0",pred,file=f2,sep='\t')
    else:
        print(olddata[0],olddata[1],"1",file=f2,sep='\t')
f.close()  
f2.close()
#1为男性，0为女性