import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.DataPrepare_zsl import load_data
from torch.utils.data import TensorDataset, DataLoader
from resnet50 import resnet50,resnet101

import os

import pandas as pd

from RN import RelationNetwork, AttributeNetwork
import scipy.io as sio
import numpy as np

from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def train_RN( Attri_nn, RN, cnn_fc, device, train_loader,  word_embeddings, optimizer_attr, optimizer_rn, scheduler_attr, scheduler_rn, epoch):

    Attri_nn.train()
    RN.train()

    Attri_nn.to(device)
    RN.to(device)
    scheduler_attr.step(epoch)
    scheduler_rn.step(epoch)

    cnn_fc.train()
    cnn_fc.to(device)

    word_embeddings = word_embeddings[:365, :]
    Acc = 0
    for batch_idx, sample in enumerate(train_loader):

        data = sample[0].to(device)
        label = sample[1].to(device).squeeze(1)

        re_label = []
        for slabel in label.cpu().numpy():
            if slabel not in re_label:
                re_label.append(slabel)

        word_embeddings = word_embeddings.to(device)
        optimizer_attr.zero_grad()
        optimizer_rn.zero_grad()


        word_embeddings2 = word_embeddings[re_label]  # select the attribute
        class_num = word_embeddings2.shape[0]   # use these labels to train in this batch
        batch_size = data.shape[0]
        data = F.relu(cnn_fc(data))
        cnn_feat = data.unsqueeze(0).repeat(class_num, 1, 1)

        semantic_feat = Attri_nn(word_embeddings2).unsqueeze(0).repeat(batch_size, 1, 1)
        cnn_feat = torch.transpose(cnn_feat, 0, 1)
        score = RN(cnn_feat, semantic_feat)
        score = score.view(-1, class_num)



        re_batch_labels = []
        re_label = np.array(re_label)
        for slabel in label.cpu().numpy():
            index = np.argwhere(re_label == slabel)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)


        mse = nn.MSELoss().to(device)
        one_hot_labels = torch.zeros(batch_size, class_num).scatter_(1, re_batch_labels.view(-1, 1), 1).to(device)
        loss = mse(score, one_hot_labels)
        loss.backward()
        optimizer_rn.step()
        optimizer_attr.step()
        _, predict_labels = torch.max(score, 1)
       # _, gt_labels = torch.max(one_hot_labels, 1)
        rewards = [1 if predict_labels[j].cpu() == re_batch_labels[j] else 0 for j in range(batch_size)]
        Acc += np.sum(rewards)
        if batch_idx % args.log_interval == 0:
                log_print = 'Train RN Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item())
                print(log_print)
                with open(args.log, 'a') as f:
                    print(log_print, file=f)
    train_acc = Acc/len(train_loader.dataset)
    print("Train Acc is %.3f"%train_acc)


    with open(args.log, 'a') as f:
        print("Train Acc is %.3f"%train_acc, file=f)


def test_RN(Attri_nn, RN, cnn_fc, device, test_loader,  word_embeddings):
    Attri_nn.eval()
    cnn_fc.eval()
    RN.eval()
    Attri_nn.to(device)
    RN.to(device)
    cnn_fc.to(device)
    total_rewards= 0
    word_embeddings = word_embeddings[365:365+40, :]
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):

            data = sample[0].to(device)

            label = sample[1].to(device).squeeze(1)
            word_embeddings = word_embeddings.to(device)

            re_label = []
            for slabel in label.cpu().numpy():
                if slabel not in re_label:
                    re_label.append(slabel)

            word_embeddings2 = word_embeddings
            class_num = word_embeddings2.shape[0]
            batch_size = data.shape[0]
            data = F.relu(cnn_fc(data))
            cnn_feat = data.unsqueeze(0).repeat(class_num, 1, 1)
            wd_feat = Attri_nn(word_embeddings2).unsqueeze(0).repeat(batch_size, 1, 1)
            cnn_feat = torch.transpose(cnn_feat, 0, 1)
            scores = RN(cnn_feat, wd_feat).view(-1, class_num)

            re_batch_labels = []
            re_label = np.array(re_label)
            for slabel in label.cpu().numpy():
                index = np.argwhere(re_label == slabel)
                re_batch_labels.append(index[0][0])
            _, predict_labels = torch.max(scores, 1)

            rewards = [1 if predict_labels[j].cpu() == label[j].long().cpu() else 0 for j in range(batch_size)]
            total_rewards += np.sum(rewards)
        num_test = len(test_loader.dataset)
        test_accuracy = total_rewards/len(test_loader.dataset)
        print("Test Acc is %.5f"%test_accuracy)
        with open(args.log, 'a') as f:
            print("the Dem accuracy is %.5f"%test_accuracy, file=f)





def ZSL_result(cnn_fc,  Attr_model, RN_model, device, test_loader, word_embeddings, save_path, classes = None, test_code = None):
    cnn_fc.eval()
    Attr_model.eval()
    RN_model.eval()
    cnn_fc.to(device)
    Attr_model.to(device)
    RN_model.to(device)
    imgpath = pd.read_csv('/home/xd133/zero-shot-gcn/round2_DatasetA_20180927/image.txt', header=None)[0]
    word_embeddings = word_embeddings[-45:, :].to(device)

    with torch.no_grad():
        pred_decoder = []
        for batch_idx, sample in tqdm(enumerate(test_loader), total=len(test_loader.dataset)):

            data = sample[0].to(device)

            batch_size = data.shape[0]
            #cnn_feat = data
            cnn_feat = F.relu(cnn_fc(data))

            wd_feat = Attr_model(word_embeddings).unsqueeze(0).repeat(batch_size, 1, 1)


            cnn_feat = cnn_feat.unsqueeze(0).repeat(45, 1, 1)
            cnn_feat = torch.transpose(cnn_feat, 0, 1)
            scores = RN_model(cnn_feat, wd_feat).view(-1, 45)

            _, predict_labels = torch.max(scores, 1)

            pred = predict_labels.cpu().numpy()[0]
            tmp = test_code

            pred_decoder.append(tmp[pred])
        preds = pd.Series(pred_decoder)
        result = pd.DataFrame([imgpath, preds]).T
        result.to_csv(save_path, sep = '\t', header=None, index=None)






if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Pytorch baseline')
    parser.add_argument('--log_interval', type=int, default=10, help='the interval of the display')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-5, help='weight_decay')
    parser.add_argument('--epoch', type=int, default=1000, help='the training epoch')
    parser.add_argument('--log', type=str, default='/home/xd133/ZJL_Fusai/log/zsl_log1019_2.txt', help='where to save the log')
    parser.add_argument('--save_path', type=str, default = '/home/xd133/ZJL_Fusai/output/cls_zsl1019.txt', help='path to save the result file')
    parser.add_argument('--test', type = bool, default=False, help='only test?')
    parser.add_argument('--epoch_decay', type=int, default = 30, help = 'decay 0.1 every epoch_decay epoches')
    args = parser.parse_args()
    device = torch.device('cuda')


    word_embedding = load_data(batch_size=args.batch_size)[0]

    word_embedding = torch.from_numpy(word_embedding).float()
    test_code = load_data(batch_size=args.batch_size)[-1]

    train_feats = sio.loadmat('/home/xd133/ZJL_Fusai/Features/train.mat')['features']
    train_labels = sio.loadmat('/home/xd133/ZJL_Fusai/Features/train.mat')['label']

    val_feats = sio.loadmat('/home/xd133/ZJL_Fusai/Features/val.mat')['features']
    val_labels = sio.loadmat('/home/xd133/ZJL_Fusai/Features/val.mat')['label']

    test_feats = sio.loadmat('/home/xd133/ZJL_Fusai/Features/test.mat')['features']


    train_feats = torch.from_numpy(train_feats)
    train_labels = torch.from_numpy(train_labels)

    val_feats = torch.from_numpy(val_feats)
    val_labels = torch.from_numpy(val_labels)

    test_feats = torch.from_numpy(test_feats)

    train_data = TensorDataset(train_feats, train_labels)
    val_data = TensorDataset(val_feats, val_labels)
    test_data = TensorDataset(test_feats)


    train_featloader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True)
    val_featloader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_featloader = DataLoader(test_data, batch_size=1, shuffle=False)



    model_Attr = AttributeNetwork(300, 1200, 2048)
    model_RN = RelationNetwork(4096, 1024)
    cnn_fc = nn.Linear(2048, 2048)

    optimizer_Attr = optim.Adam(list(model_Attr.parameters()) + list(cnn_fc.parameters()), lr=args.lr, weight_decay=args.wd)
    #optimizer_Attr = optim.Adam(model_Attr.parameters(), lr=args.lr,
    #                            weight_decay=args.wd)
    scheduler_Attr = optim.lr_scheduler.StepLR(optimizer_Attr, 200)
    optimizer_RN = optim.Adam(model_RN.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler_RN = optim.lr_scheduler.StepLR(optimizer_RN, 200)



    if args.test == False:
        for epoch in range(1, args.epoch):
            train_RN(model_Attr, model_RN, cnn_fc, device, train_featloader, word_embedding, optimizer_Attr, optimizer_RN,
                       scheduler_Attr, scheduler_RN, epoch)
            # train_RN(model_Attr, model_RN, device, train_featloader, word_embedding, optimizer_Attr,
            #          optimizer_RN, scheduler_Attr, scheduler_RN, epoch)


            test_RN(model_Attr, model_RN, cnn_fc, device, val_featloader, word_embedding)
            #test_RN(model_Attr, model_RN, device, val_featloader, word_embedding)

            torch.save(model_Attr.state_dict(), "/home/xd133/ZJL_Fusai/output_zsl1019_2/Attr{:d}.pt".format(epoch))
            torch.save(model_RN.state_dict(), "/home/xd133/ZJL_Fusai/output_zsl1019_2/RN{:d}.pt".format(epoch))
            torch.save(cnn_fc.state_dict(), "/home/xd133/ZJL_Fusai/output_zsl1019_2/Fc{:d}.pt".format(epoch))

    else:
        model_RN.load_state_dict(torch.load('/home/xd133/ZJL_Fusai/output_zsl1019_2/RN23.pt'))
        model_Attr.load_state_dict(torch.load('/home/xd133/ZJL_Fusai/output_zsl1019_2/Attr23.pt'))
        cnn_fc.load_state_dict(torch.load('/home/xd133/ZJL_Fusai/output_zsl1019_2/Fc23.pt'))
        ZSL_result(cnn_fc, model_Attr, model_RN, device, test_featloader, word_embedding, args.save_path, test_code= test_code)





