from SVHN_main import ResCNN
import torch
import torchvision
import os
from PIL import Image
import numpy as np
import time


trs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
t = '02173516'
layers = [2, 8, 16, 2]
channels = [32, 64, 128, 256]

folder = './street_data/test/'
fileorder = list(map(lambda x: int(x.replace('.jpg', '')), os.listdir(folder)))
fileorder = sorted(fileorder)
weight_path = "./log/"+str(t)+'.p'
C = ResCNN(layers, channels).cuda()
C.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage.cuda(0)))
T = time.strftime("%H%M%S")
print(T)
pics = [None for _ in range(1800)]
results = []
for e in fileorder:
    img = Image.open(folder+str(e)+'.jpg')
    # print(np.array(img).shape)
    img = trs(img).cuda()
    img = torch.unsqueeze(img, 0)
    pics[e] = img
    # print(img.shape)
p = torch.Tensor(1800, 3, 32, 32).cuda()
torch.cat(pics, out=p)
for i in range(0, 1800, 200):
    output = C(torch.autograd.Variable(p[i:i+200]).cuda())
    index = torch.argmax(output, 1)
    results+=np.array(index).tolist()

with open('./submission_'+T+'.csv', 'w+') as f:
    for e in results:
        f.write(str(int(e))+'\n')
    f.close()
