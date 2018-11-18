import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import torch
from sklearn.model_selection import train_test_split
from utils.transforms_self import *

train_path = '/home/xd133/zero-shot-gcn/semifinal_image_phase2/'
test_path = '/home/xd133/zero-shot-gcn/semifinal_image_phase2/'
'''
Only Load the image,  word embedding and label
'''


def load_data(batch_size=64,alldata=False):

    all_data = pd.read_csv(train_path + 'alltrain.txt', delimiter='\t', header=None, names=['ImageName', 'label'])



    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=43,
                                         stratify=all_data['label'])




    test_data = pd.read_csv(test_path + 'image.txt', header=None, names=['ImageName', 'label'])


    if alldata: #是否使用全部数据集训练
        train_data = all_data

    label_code = pd.Series(sorted(list(set(list(train_data['label'].values)))))


    # str -> number
    label_map = {label_code[i]: i for i in range(len(label_code))}


    train_data['label'] = train_data['label'].apply(lambda x: label_map[x])  #train label
    val_data['label'] = val_data['label'].apply(lambda x: label_map[x])      #val label



    transform_train = transforms.Compose([
        RandomRotate(angles=(-45, 45)),
        RandomHflip(),
       # ExpandBorder(size=(112, 112), resize=True),
        RandomResizedCrop(size=(299, 299),scale=(0.8, 1), ratio=(3. / 4., 4. / 3.)),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        #ExpandBorder(size=(100, 100), resize=True),
        RandomResizedCrop(size=(299, 299), scale=(0.8, 1), ratio=(3. / 4., 4. / 3.)),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    datasets = {'train': dataset(train_path + 'alltrain', train_data, transforms=transform_train),
                'val': dataset(train_path+'alltrain', val_data, transforms=transform_val),
                'test': dataset(test_path + 'test', test_data, transforms = transform_val)
                }
    dataloaders = {
    ds: DataLoader(datasets[ds], batch_size=batch_size if ds != 'test' else 1, shuffle=False if ds != 'train' else True,
                   pin_memory=True, num_workers=8, drop_last=False, collate_fn=collate_fn) for ds in datasets}

    return dataloaders




class dataset(Dataset):
    def __init__(self, imgroot, anno_pd, transforms=None, aug = False):
        self.root_path = imgroot
        self.paths = anno_pd['ImageName'].tolist()
        try:
            self.labels = anno_pd['label'].tolist()
        except:
            self.labels = anno_pd['ImageName'].tolist()  #测试集没有label则不用
        self.transforms = transforms
        self.aug = aug
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        # img = self.pil_loader(img_path)
        img =cv2.imread(img_path)
        try:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   # [h,w,3]  RGB
        except:
            pass
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.labels[item]

        return torch.from_numpy(img).float(), label

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
def collate_fn(batch):
    imgs = []
    label = []

    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label