#coding=utf-8

'''
统计数据分布情况，判断是否有不均衡的现象
'''
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# alldata = pd.read_csv('/home/xd133/ZJL_zero_shot/DatasetB_20180919/alltrain.csv', delimiter='\t', header=None)
#
# data_count = alldata[1].value_counts()
#
# data_count = data_count[sorted(data_count.index, key=lambda s: int(s[3:]))]
#
#
#
# data_count.plot('bar')
# plt.show()
#

# '''
# 判断测试集中包含多少训练集
# '''
# import pandas as pd
#
# test_imgs = pd.read_csv('/home/xd133/zero-shot-gcn/DatasetB_20180919/image.txt', header=None)
# train_imgs = pd.read_csv('/home/xd133/zero-shot-gcn/DatasetB_20180919/alltrain.csv', header=None, delimiter='\t')
#
# train_images = train_imgs[0].values
# test_images = test_imgs[0].values
#
# cnt = 0
# for train_image in train_images:
#     train_image = train_image.replace('.jpeg', '.jpg')
#     if train_image in (test_images):
#        cnt = cnt+1
# print (cnt)


'''
查看词向量情况
'''
# import pandas as pd
# embedding = '/home/xd133/zero-shot-gcn/DatasetB_20180919/class_wordembeddings.txt'
# embedding = pd.read_csv(embedding, header=None, delimiter=' ')
# em_kind = embedding[0].values
# num = len(em_kind)


'''
查看unseen label
'''
import pandas as pd

train_txt = pd.read_csv('/home/xd133/zero-shot-gcn/DatasetB_20180919/alltrain.csv', header=None, delimiter='\t')
label_list = pd.read_csv('/home/xd133/zero-shot-gcn/DatasetB_20180919/label_list.txt', header=None, delimiter='\t')

train_label = set(train_txt[1])
all_label = set(label_list[0])
test_label = pd.Series(list(train_label ^ all_label))
pass

