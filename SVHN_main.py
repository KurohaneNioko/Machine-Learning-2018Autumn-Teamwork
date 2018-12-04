import torch
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import numpy
import time
import sys

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
        shuffle= True if label=='train' else False)
    return ds, dl

# s, l = get_ds(256)
# for j in range(10):
#     t = 0
#     for i in range(len(s)):
#         temp = s[i][1]
#         if(temp==j): t+=1
#     print(j, "count=", t)

with_bias = False

class ResidualUnit(torch.nn.Module):

    def __init__(self, in_channel, out_channel, conv_step=1, input_fitter=None):
        super(ResidualUnit, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channel, out_channel, bias=with_bias,
            kernel_size=3, stride=conv_step, padding=1
        )
        self.batch_norm1 = torch.nn.BatchNorm2d(out_channel)
        self.active_func1 = torch.nn.ReLU(inplace=True)
        self.conv_2 = torch.nn.Conv2d(
            out_channel, out_channel, bias=with_bias,
            kernel_size=3, padding=1
        )
        self.batch_norm2 = torch.nn.BatchNorm2d(out_channel)
        self.input_fitter = input_fitter

        self.active_func2 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        # x: batch,
        residual = x

        out = self.conv_1(x)
        out = self.batch_norm1(out)
        out = self.active_func1(out)
        out = self.conv_2(out)
        out = self.batch_norm2(out)
        # print(out.shape, x.shape, sep='\n')
        if out.shape != residual.shape:
            residual = self.input_fitter(residual)
        out = out + residual
        out = self.active_func2(out)

        return out


class ResCNN(torch.nn.Module):

    def __init__(self, layers, channel_list, types=10):
        # layers:
        assert len(layers) == len(channel_list) #- 1
        # at first channel of pic is R,G,B, 3 channels
        super(ResCNN, self).__init__()
        self.conv_begin = torch.nn.Conv2d(
            in_channels=3, out_channels=channel_list[0],
            kernel_size=3, stride=2, padding=1, bias=with_bias
        )
        self.batch_norm = torch.nn.BatchNorm2d(channel_list[0])
        self.active_func = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )

        subnet_list = []
        last_channel = channel_list[0]
        for channel, layer in zip(channel_list, layers):
            fitter = torch.nn.Sequential(
                torch.nn.Conv2d(
                    last_channel, channel,
                    kernel_size=1, stride=2, bias=False),
                torch.nn.BatchNorm2d(channel),
            )
            subnet_list.append(
                ResidualUnit(last_channel, channel, conv_step=2, input_fitter=fitter)
            )
            for i in range(1, layer):
                subnet_list.append(
                    ResidualUnit(channel, channel, input_fitter=fitter)
                )
            last_channel = channel
        self.subnet = torch.nn.Sequential(*subnet_list)
        # print(self.subnet)
        self.avgpool = torch.nn.AvgPool2d(1, stride=4)
        self.full_connect = torch.nn.Linear(last_channel, types)

    def forward(self, x):
        # x: batch, channel, pic_dim1, pic_dim2
        x = self.conv_begin(x)
        # x: batch, channel_list[0], pic_dim1, pic_dim2
        x = self.batch_norm(x)
        x = self.active_func(x)
        x = self.maxpool(x)
        x = self.subnet(x)

        x = self.avgpool(x)
        x = self.full_connect(x.squeeze())
        return x


layers = [2, 2, 2, 2]
channels = [32, 64, 128, 256]
#training
batch_size = 256
epmax = 60
wd = 1e-5
# dynamic lr
init_lr = 1e-3
step_sz = 20
gamma = 0.1


def train(train_loader, test_loader):
    CNN = ResCNN(layers, channels).cuda()
    # from torchvision.models.resnet import BasicBlock
    # CNN = torchvision.models.ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).cuda()
    # print(CNN)
    lossfunc = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(CNN.parameters(), lr=init_lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_sz, gamma=gamma)
    tr_ls = []
    te_ls = []
    te_acc = []
    te_min_index = 0
    te_acc_max_index = 0
    T = time.strftime("%H%M%S")
    print(time.strftime("%H:%M:%S"))
    logfile = open('./log/'+T+'.txt', 'w+')
    print('layers =', str(layers), 'channels =', str(channels),
        'batch_size =', batch_size, 'ep_max =', epmax,
        'weight_decay =', wd, 'init_lr=', init_lr,
        'step_size=', step_sz, 'gamma=', gamma, file=logfile)
    logfile.flush()
    weight_file_count = 0
    for epoch in range(epmax):
        # print('current epoch = %d' % (epoch+1))
        train_loss = 0.0
        test_total_loss = 0.0
        train_batch_cnt, validation_batch_cnt = 0, 0
        # T = time.time()
        torch.cuda.empty_cache()
        for key, value in train_loader:
            train_batch_cnt += 1
            key = torch.autograd.Variable(key).cuda()
            value = torch.autograd.Variable(value).cuda()
            optimizer.zero_grad()
            outputs = CNN(key).squeeze()
            loss = lossfunc(outputs, value)
            loss.backward()
            optimizer.step()
            train_loss += loss
            # if i % 100 == 0:
            # print('current loss = %.5f' % loss.item())
        scheduler.step()
        torch.cuda.empty_cache()
        with torch.no_grad():
            correct_rate = 0
            for key, value in test_loader:
                validation_batch_cnt += 1
                key = torch.autograd.Variable(key).cuda()
                value = torch.autograd.Variable(value).cuda()
                outputs = CNN(key).squeeze()
                # for e in outputs:
                #     print(e)
                # print('')
                loss = lossfunc(outputs, value)
                # calc accuracy
                index = torch.argmax(outputs, 1)
                correct_rate += float(torch.sum(torch.eq(index, value)))/float(len(outputs))
                test_total_loss += loss
        te_acc.append(correct_rate / validation_batch_cnt)
        tr_ls.append(train_loss.data / train_batch_cnt)
        te_ls.append(test_total_loss.data / validation_batch_cnt)
        # print(time.time()-T)
        # if int((numpy.where(te_ls == numpy.min(te_ls)))[0]) + 1 == epoch + 1 and numpy.min(te_ls) < 0.223:
        #     torch.save(CNN.state_dict(), './weight_' + str(hs) + '_' + str(hl) + '_' + str(int(epoch + 1)) + '_' + str(
        #         numpy.min(te_ls)) + '.p')

        if te_ls[-1] < te_ls[te_min_index]:
            te_min_index = len(te_ls) - 1
            if (epoch + 1 >= 10):
                weight_file_count += 1
                torch.save(CNN.state_dict(), './log/' + T + str(weight_file_count) + '.p')
        if te_acc[-1] > te_acc[te_acc_max_index]:
            te_acc_max_index = len(te_acc) - 1
            if(epoch+1>=10):
                weight_file_count += 1
                torch.save(CNN.state_dict(), './log/' + T + str(weight_file_count) + '.p')
        rec = (
            'epoch=' + str(epoch + 1) +
            ' BestLoss: ' + str(numpy.min(te_ls)) + ' @' + str(te_min_index + 1) +
            ' BestAcc: ' + str(te_acc[te_acc_max_index])
        )
        print(rec)
        print(rec, file=logfile)
        logfile.flush()
        # print(te_ls)

        if epoch > 0 and (epoch+1) % 10 is 0:
            print(time.strftime("%H:%M:%S"))
            epoches = range(1, epoch+2)
            plt.plot(epoches, tr_ls, label='Trainning Loss', color='blue')
            plt.plot(epoches, te_ls, label='Validation Loss', color='red')
            plt.plot(epoches, te_acc,label='Validation Acc', color='purple')
            plt.grid()
            plt.xlabel('epoches')
            plt.ylabel('Loss')
            plt.legend()
            #plt.savefig(str(hs)+'_'+str(hl)+'.png')
            #plt.close('all')
            # if (int((numpy.where(te_ls == numpy.min(te_ls)))[0]) + 1 == epoch+1 and epoch>14) or epoch >=22:
            plt.show()
            # else:
            #     plt.close('all')

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
    plt.plot(epoches, te_acc, label='Validation Acc', color='purple')
    plt.xlabel('epoches')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig(T+'.png')
    # plt.close('all')
    plt.show()
    rec = (
        'layers =', str(layers), 'channels =', str(channels),
        'batch_size =', batch_size, 'ep_max =', epmax,
        'weight_decay =', wd, 'init_lr=', init_lr,
        'step_size=', step_sz, 'gamma=', gamma,
        str(int((numpy.where(te_ls == numpy.min(te_ls)))[0]) + 1) + ',' +
        str(numpy.min(tr_ls)) + ',' + str(int((numpy.where(tr_ls == numpy.min(tr_ls)))[0]) + 1)
    )
    print(rec, file=logfile)
    print(rec)
    logfile.flush()
    # torch.save(CNN.state_dict(), './single_weight_'+str(hs)+'_'+str(hl)+'.p')
    # return CNN

if __name__ == '__main__':
    trainset, trainloder = get_ds(batch_size)
    testset, testloder = get_ds(batch_size, 'test')
    train(trainloder, testloder)