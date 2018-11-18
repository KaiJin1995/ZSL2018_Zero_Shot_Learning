本项目主要时借鉴了RelationNetwork的方法，实现对未知类别的分类。
本次数据集采用之江杯的零样本比赛的数据集。
其中，本代码总共包含两个部分：分类部分和zero shot部分。分类部分采用CNN网络实现对图像特征的提取。zero-shot部分使用RelationNetwork完成对未知样本的分类。
因此，本代码分别写了分类程序和zero-shot部分的程序。
分类程序： main_cls.py
零样本程序：main_zsl.py


依赖环境： python3.6   pytorch0.4.0

操作方法：
修改好路径，执行main_cls.py，完成对图像的分类，之后运行utils中的ExtractFeature，将图像的特征单独提取出来用.mat格式保存。

之后用提取的特征训练zero-shot部分，修改相关路径，执行main_zsl.py，即可实现对零样本的分类。


