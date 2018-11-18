import argparse
import torch
from utils.DataPrepare_ex import load_data
from resnet50 import resnet50,resnet101,resnet34
import os
import scipy.io as sio
import numpy as np



os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def trainFeature(model, train_loader, device):
    model.eval()
    feats = torch.empty(len(train_loader.dataset), 2048)
    labels = torch.empty(len(train_loader.dataset), 1)
    with torch.no_grad():
        for batch_idx, sample in enumerate(train_loader):
            data = sample[0].to(device)
            label = torch.from_numpy(np.array(sample[1])).to(device)
            cnn_feat = model(data)[0]
            feats[batch_idx*64:(batch_idx+1)*64, :] = cnn_feat
            labels[batch_idx*64:(batch_idx+1)*64, 0] = label
        sio_content = {"features":feats.numpy(), "label":labels.numpy()}

        sio.savemat("/home/xd133/ZJL_Fusai/Feature_1029/train.mat", sio_content)


def testFeature(model, test_loader, device):
    model.eval()
    feats = torch.empty(len(test_loader.dataset), 2048)

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data = sample[0].to(device)
            cnn_feat = model(data)[0]
            feats[batch_idx*64:(batch_idx+1)*64, :] = cnn_feat

            print("the batchidx is %d" % batch_idx)
        sio_content = {"features":feats.numpy()}

        sio.savemat("/home/xd133/ZJL_Fusai/Feature_1029/test.mat", sio_content)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch baseline')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test', type=bool, default=False, help='only test?')
    parser.add_argument('--weights', type=str, default='/home/xd133/ZJL_Fusai/output_1027_3_3/cls99.pt', help='pretrained model path')

    args = parser.parse_args()
    device = torch.device('cuda')


    train_loader = load_data(batch_size=args.batch_size, alldata=True)['train']
    val_loader = load_data(batch_size=args.batch_size, alldata=True)['val']
    test_loader = load_data(batch_size=args.batch_size, alldata=True)['test']


    model = resnet101(num_classes=365).to(device)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    model.fc = torch.nn.Linear(model.fc.in_features, 365)
    model.to(device)
    model.load_state_dict(torch.load(args.weights))
    if args.test:   #是否是训练集或者验证集的特征
        testFeature(model, test_loader, device)
    else:
        trainFeature(model, train_loader, device)


