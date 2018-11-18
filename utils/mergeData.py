import pandas as pd
'''
合并txt数据集
'''

# txt0813 = '/home/xd133/zero-shot-gcn/DatasetA_train_20180813/train.txt'
# txt0919 = '/home/xd133/ZJL_zero_shot/DatasetB_20180919/train.txt'
#
#
# pd0813 = pd.read_csv(txt0813, header=None, delimiter='\t')
# pd0919 = pd.read_csv(txt0919, header=None, delimiter='\t')
#
#
# pd_all = pd0813.append(pd0919)
# pd_all.index = range(len(pd_all))
# pd_all.to_csv('/home/xd133/ZJL_zero_shot/DatasetB_20180919/alltrain.csv', sep = '\t', header=None, index=None)
#

# '''
# 合并word embeddings
#
# '''
# new_wd = pd.read_csv('/home/xd133/zero-shot-gcn/round2_DatasetA_20180927/class_wordembeddings.txt', header=None, delimiter=' ')
# old_wd = pd.read_csv('/home/xd133/zero-shot-gcn/DatasetB_20180919/class_wordembeddings.txt', header=None, delimiter=' ')
#
# all_wd = old_wd.append(new_wd)
# all_wd.to_csv('/home/xd133/zero-shot-gcn/round2_DatasetA_20180927/class_wordembeddings_all.txt', header=None, sep=' ', index=None)
#


# #生成新的词向量
#
#
# import csv
# wdnew = pd.read_csv('/home/xd133/word2vec-api-master/crawl-300d-2M.vec', header=None, delimiter=' ', quoting=csv.QUOTE_NONE)
# label_list = pd.read_csv('/home/xd133/zero-shot-gcn/DatasetA_test_20180813/DatasetA_test/label_list.txt', header=None, delimiter='\t').iloc[-40:]
# wdnew2 = wdnew[wdnew[0].isin(list(label_list[1].values))]
# wdnew2 = label_list.merge(wdnew2, left_on=1, right_on=0)
# wdnew2 = wdnew2.drop([1, '1_x', '0_y'], axis = 1)
#
# #label_list = pd.read_csv('/home/xd133/zero-shot-gcn/DatasetA_test_20180813/DatasetA_test/new_embed.txt', header=None, delimiter=' ')
#
#
# wdnew2.to_csv('/home/xd133/zero-shot-gcn/DatasetA_test_20180813/DatasetA_test/new_embed_crawl300.txt', header=None, sep=' ', index=None)




import csv
wdnew = pd.read_csv('/home/xd133/word2vec-api-master/crawl-300d-2M.vec', header=None, delimiter=' ', quoting=csv.QUOTE_NONE)
label_list = pd.read_csv('/home/xd133/zero-shot-gcn/semifinal_image_phase2/label_list.txt', header=None, delimiter='\t')
wdnew2 = wdnew[wdnew[0].isin(list(label_list[1].values))]
wdnew2 = label_list.merge(wdnew2, left_on=1, right_on=0)
wdnew2 = wdnew2.drop([1, '1_x', '0_y'], axis = 1)

#label_list = pd.read_csv('/home/xd133/zero-shot-gcn/DatasetA_test_20180813/DatasetA_test/new_embed.txt', header=None, delimiter=' ')


wdnew2.to_csv('/home/xd133/zero-shot-gcn/semifinal_image_phase2/class_wordembeddingscrawl300.txt', header=None, sep=' ', index=None)








