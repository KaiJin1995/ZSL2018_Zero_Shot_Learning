import pandas as pd
import random

# train_dataLabel = pd.read_csv('/home/xd133/zero-shot-gcn/DatasetA_train_20180813/train.txt', delimiter='\t', header= None)
# All_label = list(set(list(train_dataLabel.iloc[:, 1].values)))
# random.shuffle(All_label)
# train_label = All_label[:int(len(All_label)*4/5)]
# val_label = All_label[int(len(All_label)*4/5):]
#
# train_f = train_dataLabel[pd.Series([train_dataLabel[1][i] in train_label for i in range(len(train_dataLabel[1]))])]
# val_f = train_dataLabel[pd.Series([train_dataLabel[1][i] in val_label for i in range(len(train_dataLabel[1]))])]
#
# train_f.to_csv('/home/xd133/zero-shot-gcn/DatasetA_train_20180813/Train.txt', sep='\t', header=None, index = None)
# val_f.to_csv('/home/xd133/zero-shot-gcn/DatasetA_train_20180813/Val.txt', sep='\t', header=None, index = None)


train_dataLabel = pd.read_csv('/home/xd133/zero-shot-gcn/DatasetA_train_20180813/train.txt', delimiter='\t', header= None)
train_dataLabel = train_dataLabel.sample(frac = 1).reset_index(drop=True)

Train = train_dataLabel.iloc[:int(len(train_dataLabel)*4/5), :]
Val = train_dataLabel.iloc[int(len(train_dataLabel)*4/5):, :]
Train.to_csv('/home/xd133/zero-shot-gcn/DatasetA_train_20180813/Train.txt', sep='\t', header=None, index = None)
Val.to_csv('/home/xd133/zero-shot-gcn/DatasetA_train_20180813/Val.txt', sep='\t', header=None, index = None)
