import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split

train_path = '/home/xd133/zero-shot-gcn/round2_DatasetA_20180927/'
test_path = '/home/xd133/zero-shot-gcn/round2_DatasetA_20180927/'

'''
Only Load the image,  word embedding and label
'''


def load_data(batch_size=64):
    label_list = pd.read_csv(train_path + 'label_list_all.txt', delimiter='\t', header=None)
    embedding = pd.read_csv(train_path + 'class_wordembeddings_all.txt', delimiter=' ', header=None)
    all_data = pd.read_csv(train_path + 'alltrain.csv', delimiter='\t', header=None)


    # ''' train all data, or commit it'''
    train_data = all_data


    attribute = pd.read_csv(test_path + 'attributes_per_class.txt', delimiter='\t', header=None)

    embedding = label_list.merge(embedding, left_on=1, right_on=0)

    embedding = embedding.drop([1, '1_x', '0_y'], axis=1)

    subLabel = set(list(train_data[1].values))
    allLabel = set(list(label_list[0]))
    label_code = pd.Series(sorted(list(set(list(train_data[1].values)))))
    remainLabel = allLabel ^ subLabel
    remainLabel = sorted(list(remainLabel))
    remain_code45 = pd.Series(remainLabel[-45:])




    All_code = pd.Series(sorted(list(set(list(train_data[1].values)))) + remainLabel)


    label_map = {label_code[i]: i for i in range(len(label_code))}
    All_map = {All_code[i]: i for i in range(len(All_code))}


    train_data[1] = train_data[1].apply(lambda x: label_map[x])

    embedding['0_x'] = embedding['0_x'].apply(lambda x: All_map[x])
    embedding.sort_values(by='0_x', inplace=True)
    embedding.reset_index(inplace=True, drop=True)
    embedding = embedding.iloc[:, 1:].values
    attribute[0] = attribute[0].apply(lambda x: All_map[x])
    attribute.sort_values(by=0, inplace=True)
    attribute.reset_index(inplace=True, drop=True)
    attribute = attribute.iloc[:, 1:].values

    return embedding, label_code, attribute, remain_code45

class Imagedata(Dataset):

    def __init__(self, rootpath, imgpath, Label=None, attribute=None, label_code=None, animal_code=None,
                 trans_code=None, plant_code=None, other_code=None, word_embeddings=None, transforms=None,
                 all_code=None):
        self.rootpath = rootpath
        self.transforms = transforms
        self.imgpath = imgpath
        self.Label = Label

        self.label_code = label_code
        self.animal_code = animal_code
        self.trans_code = trans_code
        self.plant_code = plant_code
        self.other_code = other_code
        self.all_code = all_code
        self.word_embeddings = word_embeddings

    def __len__(self):
        return len(self.imgpath)

    def __getitem__(self, idx):
        if self.Label is not None:
            sample = {
                'image': self.transforms(Image.open(os.path.join(self.rootpath, self.imgpath[idx])).convert('RGB')),
                'label': self.Label[idx]}
        else:
            sample = {
                'image': self.transforms(Image.open(os.path.join(self.rootpath, self.imgpath[idx])).convert('RGB'))}

        return sample

