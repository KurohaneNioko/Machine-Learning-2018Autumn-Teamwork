import torch
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import numpy

def get_ds(batch_size, label='train'):
    if label == 'train':
        trs = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        trs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    ds = torchvision.datasets.SVHN(
        './SVHN_official_data',
        label,
        transform=trs,
        download=False)
    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        num_workers=6,
        shuffle=True)
    return ds, dl

# s, l = get_ds(256)
# for j in range(10):
#     t = 0
#     for i in range(len(s)):
#         temp = s[i][1]
#         if(temp==j): t+=1
#     print(j, "count=", t)

class ResidualUnit(torch.nn.Module):

    def __init__(self, in_channel, out_channel, conv_step=1):
        super(ResidualUnit, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channel, out_channel,
            kernel_size=3, stride=conv_step, padding=1
        )
        self.batch_norm = torch.nn.BatchNorm2d(out_channel)
        self.active_func = torch.nn.ReLU(inplace=True)
        self.conv_2 = torch.nn.Conv2d(
            out_channel, out_channel,
            kernel_size=3, stride=conv_step, padding=1
        )

    def forward(self, x):
        residual = x
        out = self.conv_1(x)
        out = self.batch_norm(out)
        out = self.active_func(out)
        out = self.conv_2(out)
        out = self.batch_norm(out)
        out = out + residual
        return self.active_func(out)


class ResCNN(torch.nn.Module):

    def __init__(self, layers, channel_list, types=10):
        # layers:
        assert len(layers) - len(channel_list) == -1
        # at first channel of pic is R,G,B, 3 channels
        self.conv_begin = torch.nn.Conv2d(
            in_channels=3, out_channels=channel_list[0],
            kernel_size=4, stride=1, padding=1
        )
        self.batch_norm = torch.nn.BatchNorm2d(channel_list[0])
        self.active_func = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1
        )

        subnet_list = []
        last_channel = channel_list[0]
        for channel, layer in zip(channel_list[1:], layers):
            subnet_list.append(
                ResidualUnit(last_channel, channel, conv_step=2)
            )
            for i in range(1, layer):
                subnet_list.append(
                    ResidualUnit(channel, channel, conv_step=2)
                )
            last_channel = channel
        self.subnet = torch.nn.Sequential(*subnet_list)

        self.avgpool = torch.nn.AvgPool2d(4, stride=1)
        self.full_connect = torch.nn.Linear(last_channel, types)

    def forward(self, x):
        x = self.conv_begin(x)
        x = self.batch_norm(x)
        x = self.active_func(x)
        x = self.maxpool(x)
        x = self.subnet(x)
        x = self.avgpool(x)
        x = self.full_connect(x.squeeze())
        return x

layers = [2, 2, 2]
channels = [16, 32, 64, 128]
#training
epmax = 700
wd = 1e-5
# dynamic lr
init_lr = 1e-3
step_sz = 70
gamma = 0.1
def train(train_loader, test_loader):
    CNN = ResCNN(layers, channels)
    lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(CNN.parameters(), lr=init_lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_sz, gamma=gamma)
    tr_ls = []
    te_ls = []
    T = 0.0
    for epoch in range(epmax):
        # print('current epoch = %d' % (epoch+1))
        train_loss = 0.0
        test_total_loss = 0.0
        train_batch_cnt, validation_batch_cnt = 0, 0
        # T = time.time()
        for key, value in train_loader:
            train_batch_cnt += 1
            key = torch.autograd.Variable(key)
            value = torch.autograd.Variable(value)
            optimizer.zero_grad()
            outputs = CNN(key).squeeze()[:, -1]
            loss = lossfunc(outputs, value)
            loss.backward()
            optimizer.step()
            train_loss += loss
            # if i % 100 == 0:
            # print('current loss = %.5f' % loss.item())
        with torch.no_grad():
            for key, value in test_loader:
                validation_batch_cnt += 1
                outputs = CNN(key).squeeze()[:, -1]
                loss = lossfunc(outputs, value)
                test_total_loss += loss
        tr_ls.append(train_loss.data / train_batch_cnt)
        te_ls.append(test_total_loss.data / validation_batch_cnt)
        # print(time.time()-T)
        if int((numpy.where(te_ls == numpy.min(te_ls)))[0]) + 1 == epoch + 1 and numpy.min(te_ls) < 0.223:
            torch.save(CNN.state_dict(), './weight_' + str(hs) + '_' + str(hl) + '_' + str(int(epoch + 1)) + '_' + str(
                numpy.min(te_ls)) + '.p')
        print(str(numpy.min(te_ls)) + ',' + str(int((numpy.where(te_ls == numpy.min(te_ls)))[0]) + 1))

        if epoch > 0:
            epoches = range(1, epoch+2)
            plt.plot(epoches, tr_ls, label='Trainning Loss', color='blue')
            plt.plot(epoches, te_ls, label='Validation Loss', color='red')
            #plt.title('Loss')
            plt.xlabel('epoches')
            plt.ylabel('Loss')
            plt.legend()
            #plt.savefig(str(hs)+'_'+str(hl)+'.png')
            #plt.close('all')
            if (int((numpy.where(te_ls == numpy.min(te_ls)))[0]) + 1 == epoch+1 and epoch>14) or epoch >=22:
                plt.show()
            else:
                plt.close('all')

    '''
    with open(logfilename, 'a+') as f:
        tr_ls = numpy.array(tr_ls)
        te_ls = numpy.array(te_ls)
        f.write(str(hs)+','+str(hl)+','+str(numpy.min(te_ls))+','+str(int((numpy.where(te_ls==numpy.min(te_ls)))[0])+1)+','+ 
                str(numpy.min(tr_ls))+','+str(int((numpy.where(tr_ls==numpy.min(tr_ls)))[0])+1))
        f.write('\n')
        f.close()
    '''
    epoches = range(1, epmax + 1)
    plt.plot(epoches, tr_ls, label='Trainning Loss', color='blue')
    plt.plot(epoches, te_ls, label='Validation Loss', color='red')
    plt.xlabel('epoches')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig(str(hs)+'_'+str(hl)+'.png')
    # plt.close('all')
    plt.show()
    print(str(hs) + ',' + str(hl) + ',' + str(numpy.min(te_ls)) + ',' + str(
        int((numpy.where(te_ls == numpy.min(te_ls)))[0]) + 1) + ',' +
          str(numpy.min(tr_ls)) + ',' + str(int((numpy.where(tr_ls == numpy.min(tr_ls)))[0]) + 1))
    # torch.save(CNN.state_dict(), './single_weight_'+str(hs)+'_'+str(hl)+'.p')
    # return CNN