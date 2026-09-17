# 数据集目录结构

## datasets/content/
放灰度医学图像（待着色的源图像）

来源: Harvard Whole Brain Atlas
  http://www.med.harvard.edu/AANLIB/home.html

应包含的5大类病例:
  - Normal Brain (正常大脑): MR T1/T2/PD, CT, PET
  - Cerebrovascular Disease (脑血管疾病): Stroke, Aneurysm 等
  - Brain Tumors (脑肿瘤): Glioma, Meningioma 等
  - Degenerative Disease (退行性疾病): Alzheimer's, Huntington's 等
  - Inflammatory/Infectious (感染/炎症): MS, AIDS 等

论文中总计筛选出 2149 张切片。每张切片存为 .jpg 或 .png。
图像会被 train.py 自动转为灰度再输入模型（RGB 3通道，但像素值 r=g=b）。

## datasets/style/
放真实人体彩色切片图像（参考颜色源）

来源: NLM Visual Human Project (VHP)
  https://www.nlm.nih.gov/research/visible/photos.html

论文中仅使用一张男性头部横断面彩色冷冻切片作为参考。
图像需为 RGB 彩色 .jpg/.png。
