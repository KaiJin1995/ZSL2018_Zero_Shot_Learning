import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split


train_path = '/home/xd133/zero-shot-gcn/round2_DatasetA_20180927/'
val_path = '/home/xd133/zero-shot-gcn/round2_DatasetA_20180927/'
test_path ='/home/xd133/zero-shot-gcn/round2_DatasetA_20180927/'


'''
Only Load the image,  word embedding and label
'''

def load_data(batch_size = 64):

    label_list = pd.read_csv(train_path + 'label_list_all.txt', delimiter='\t', header = None)
    embedding = pd.read_csv(train_path + 'class_wordembeddings_all.txt', delimiter=' ', header = None)
    all_data = pd.read_csv(train_path + 'alltrain.csv', delimiter='\t', header = None)

    # classes = np.load('/home/xd133/ZF2/test/class.npy').tolist()
    # animal = classes[0]
    # trans = classes[1]
    # plant = classes[2]
    # others = classes[3]

   # train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=43,
   #                                     stratify=all_data[1])
    val_data = pd.read_csv('/home/xd133/ZJL_Fusai/Onlytest.txt', delimiter='\t', header = None)
    # ''' train all data, or commit it'''
    train_data = all_data



    #val_data = pd.read_csv(train_path + 'Val.txt', delimiter='\t', header = None)
    test_data = pd.read_csv(test_path + 'image.txt', header = None)
    #test_data_new = pd.read_csv('/home/xd133/ZJL_zero_shot/output/testLabel.txt', header = None, delimiter='\t')


    attribute = pd.read_csv(test_path + 'attributes_per_class.txt', delimiter='\t', header=None)
  #  m = embedding.copy()
    embedding = label_list.merge(embedding, left_on=1, right_on=0)


    #embedding2 = embedding.merge(attribute, left_on='0_x',right_on=0)
   # a = embedding2
    embedding = embedding.drop([1, '1_x', '0_y'], axis=1)

    
    subLabel = set(list(train_data[1].values))
    label_code = pd.Series(sorted(list(set(list(train_data[1].values)))))

    allLabel = set(list(label_list[0]))

    remainLabel = allLabel^subLabel
    remainLabel = sorted(list(remainLabel))
    All_code = pd.Series(sorted(list(set(list(train_data[1].values)))) + remainLabel)


    remain_code = pd.Series(remainLabel)
    remain_code45 = pd.Series(remainLabel[-45:])

    #remain_code40 = pd.Series(remainLabel[40:])
    #remain_code40 = pd.Series(remainLabel[:40])
    # animal_code = remain_code40[remain_code40.isin(animal)]
    # trans_code = remain_code40[remain_code40.isin(trans)]
    # plant_code = remain_code40[remain_code40.isin(plant)]
    # other_code = remain_code40[remain_code40.isin(others)]
    # animal_code.reset_index(inplace=True, drop=True)
    # trans_code.reset_index(inplace=True, drop=True)
    # plant_code.reset_index(inplace=True, drop=True)
    # other_code.reset_index(inplace=True, drop=True)




    val_code = pd.Series(remainLabel[:40])
    # str -> number
    label_map = {label_code[i]: i for i in range(len(label_code))}
    All_map = {All_code[i]: i for i in range(len(All_code))}

    remain_map ={remain_code[i]: i for i in range(len(remain_code))}

    val_map = {val_code[i]: i for i in range(len(val_code))}
    train_data[1] = train_data[1].apply(lambda x: label_map[x])

    val_data[1] = val_data[1].apply(lambda x: val_map[x])
    #val_data[1] = val_data[1].apply(lambda x: val_map[x])
    #test_data_new[1] = test_data_new[1].apply(lambda x: All_map[x])
   # a = embedding.copy()
    embedding['0_x'] = embedding['0_x'].apply(lambda x: All_map[x])
    #b = embedding.copy()
    embedding.sort_values(by='0_x', inplace=True)
   # c = embedding.copy()
    embedding.reset_index(inplace=True, drop=True)
    #d = embedding.copy()
    embedding = embedding.iloc[:,  1:].values
    attribute[0] = attribute[0].apply(lambda x: All_map[x])
    attribute.sort_values(by = 0, inplace = True)
    attribute.reset_index(inplace=True, drop=True)
    attribute = attribute.iloc[:, 1:].values




    #all_attr = np.concatenate([embedding, attribute], axis=1)
    all_attr = embedding
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(0.5, 0.5, 0.5, 0.1),
        # transforms.RandomRotation(45),
        transforms.RandomResizedCrop(64,  (0.8, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    transform_val = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])



    datasets ={'train': Imagedata(train_path+'alltrain', train_data[0].values, train_data[1].values, attribute, transforms = transform_train),
               # 'val': Imagedata('/home/xd133/zero-shot-gcn/DatasetA_test_20180813/DatasetA_test/test/', val_data[0].values, val_data[1].values, attribute, transforms = transform_val, label_code = val_code,
               #                  animal_code=animal_code, trans_code=trans_code,plant_code=plant_code, other_code=other_code,all_code = [animal_code, trans_code, plant_code, other_code]
               #                  ),
               'val': Imagedata('/home/xd133/zero-shot-gcn/DatasetA_test_20180813/DatasetA_test/test/', val_data[0].values, val_data[1].values, attribute,
                                  transforms=transform_val),
               # 'val': Imagedata(train_path+'alltrain',
               #                  val_data[0].values, val_data[1].values, attribute,
               #                  transforms=transform_val),
               # 'test': Imagedata(test_path+'test/', test_data[0].values, attribute=attribute, label_code = remain_code40, animal_code =animal_code, trans_code = trans_code,
               #                   plant_code = plant_code, other_code= other_code,transforms = transform_val, all_code = [animal_code, trans_code, plant_code, other_code])}
               'test': Imagedata(test_path + 'test/', test_data[0].values, attribute=attribute,
                                 label_code=remain_code45,  transforms=transform_val)}
                                 #all_code=[animal_code, trans_code, plant_code, other_code])}
    dataloaders = {ds:DataLoader(datasets[ds], batch_size=batch_size if ds != 'test' else 1, shuffle=False if ds != 'train' else True, pin_memory=True, num_workers=8, drop_last=False) for ds in datasets}

    return dataloaders, embedding, label_code, attribute, all_attr




class Imagedata(Dataset):

    def __init__(self, rootpath, imgpath, Label=None, attribute = None, label_code=None, animal_code = None, trans_code = None, plant_code = None, other_code = None, word_embeddings = None, transforms=None, all_code=None):
        self.rootpath = rootpath
        self.transforms = transforms
        self.imgpath = imgpath
        self.Label = Label




        # self.attr_species = attribute[:, :6]
        # self.attr_color = attribute[:, 6:14]
        # self.attr_features = attribute[:, 14:18]
        # self.attr_use = attribute[:, 18:25]
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
            # sample = {'image': self.transforms(Image.open(os.path.join(self.rootpath, self.imgpath[idx])).convert('RGB')), 'label': self.Label[idx], 'species':self.attr_species[self.Label[idx]],
            #           'color': self.attr_color[self.Label[idx]], 'features':self.attr_features[self.Label[idx]], 'use': self.attr_use[self.Label[idx]]}
            #sample['word_embeddings'] = self.word_embeddings[sample['label']]
            sample = {
                'image': self.transforms(Image.open(os.path.join(self.rootpath, self.imgpath[idx])).convert('RGB')),
                'label': self.Label[idx]}
        else:
            sample = {'image': self.transforms(Image.open(os.path.join(self.rootpath, self.imgpath[idx])).convert('RGB'))}


        return sample



