import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.DataPrePare_cls import load_data
from resnet50 import resnet50, resnet101, resnet34
from torch.optim import lr_scheduler
import numpy as np
import os
from inceptionV4 import inceptionv4
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TrainAcc = 0

def train(args, model, device, train_loader, optimizer, epoch, exp_lr_scheduler):
    model.train()
    correct = 0
    for batch_idx, sample in enumerate(train_loader):
        data = sample[0].to(device)
        label = torch.from_numpy(np.array(sample[1])).to(device)

        optimizer.zero_grad()
        output = model(data)
        pred = output[1].max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()
        loss = F.cross_entropy(output[1], label)

        lossall = loss
        lossall.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            log_print = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLr: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), lossall.item(), exp_lr_scheduler.get_lr())
            print(log_print)
            with open(args.log, 'a') as f:
                print(log_print, file=f)

    trainAccuracy = correct / len(train_loader.dataset)
    print_log = 'Train Acc is %.2f' %trainAccuracy
    print(print_log)
    with open(args.log, 'a') as f:
        print(print_log, file=f)

maxAccuracy = 0

def test(args, model, device, test_loader, epoch):
    global maxAccuracy
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            data = sample[0].to(device)
            label = torch.from_numpy(np.array(sample[1])).to(device)

            output = model(data)
            test_loss += F.cross_entropy(output[1], label) # sum up batch loss
            pred = output[1].max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    testAccuracy = correct/len(test_loader.dataset)

    if maxAccuracy < testAccuracy:
        maxAccuracy = testAccuracy
        torch.save(model.state_dict(), '/home/xd133/ZJL_Fusai/output_1029_3/Valoutput{:d}Acc{:d}Epoch.pt'.format(int(testAccuracy*100), epoch))
        print('maxAccuracy is %.2f, save model' %maxAccuracy)
    print_log = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    print(print_log)
    with open(args.log, 'a') as f:
        print(print_log, file=f)




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Pytorch baseline')
    parser.add_argument('--log_interval', type=int, default=10, help='the interval of the display')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--epoch', type=int, default=100, help='the training epoch')
    parser.add_argument('--log', type=str, default='/home/xd133/ZJL_Fusai/log/cls_InceptionV4_1029.txt', help='where to save the log')

    args = parser.parse_args()
    device = torch.device('cuda')

    train_loader = load_data(batch_size=args.batch_size,alldata=False)['train']
    val_loader = load_data(batch_size=args.batch_size)['val']


   # model =resnet101(pretrained=False)
    model = inceptionv4(pretrained=False)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    #model.fc = torch.nn.Linear(model.fc.in_features, 365)
    model.last_linear = torch.nn.Linear(model.last_linear.in_features, 365)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr = args.lr, weight_decay=args.wd, momentum=args.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    for epoch in range(1, args.epoch):
        exp_lr_scheduler.step(epoch)

        train(args, model, device, train_loader, optimizer, epoch, exp_lr_scheduler) #训练
        test(args, model, device, val_loader, epoch) #训练

        torch.save(model.state_dict(), '/home/xd133/ZJL_Fusai/output_1029_3/cls%d.pt' % epoch)




