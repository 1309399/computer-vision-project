# 目标检测+目标跟踪+单目测距+姿态识别+车道线识别+车牌识别+A star算法+车辆跟踪与测距等视觉项目
# 目标检测+目标跟踪+单目测距+姿态识别+车道线识别+车牌识别+A star算法+车辆跟踪与测距等视觉项目


具体教程见链接：
联系方式：qq1309399183


# 图像分类

**教程博客_传送门链接:[链接](https://blog.csdn.net/ALiLiLiYa/article/details/127454333)**
在本教程中，您将学习如何使用迁移学习训练卷积神经网络以进行图像分类。您可以在 cs231n 上阅读有关迁移学习的更多信息。
本文主要目的是教会你如何自己搭建分类模型，耐心看完，相信会有很大收获。废话不多说，直切主题…
首先们要知道深度学习大都包含了下面几个方面：
1.加载（处理）数据
2.网络搭建
3.损失函数（模型优化）
4 模型训练和保存
把握好这些主要内容和流程，基本上对分类模型就大致有了个概念。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2af7bbdf024545cdbd862f70c14a267b.png)

# 目标检测

## 交通标志识别

本项目是一个基于 OpenCV 的交通标志检测和分类系统，可以在视频中实时检测和分类交通标志。检测阶段使用图像处理技术，在每个视频帧上创建轮廓并找出其中的所有椭圆或圆形。它们被标记为交通标志的候选项。
**教程博客_传送门链接------->[交通标志识别](https://blog.csdn.net/ALiLiLiYa/article/details/129468675)**
![在这里插入图片描述](https://img-blog.csdnimg.cn/be65a5bce82745a89d0717be62d036d0.png)

## 表情识别、人脸识别

面部情绪识别（FER）是指根据面部表情识别和分类人类情绪的过程。通过分析面部特征和模式，机器可以对一个人的情绪状态作出有根据的推断。这个面部识别的子领域高度跨学科，涉及计算机视觉、机器学习和心理学等领域的知识。

**教程博客_传送门链接------->[表情识别](https://blog.csdn.net/ALiLiLiYa/article/details/132795491)**
![在这里插入图片描述](https://img-blog.csdnimg.cn/c6d044aeefb14e55bb9ff95a44ac9223.png)

## 疲劳检测

瞌睡经常发生在汽车行驶的过程中，该行为害人害己，如果有一套能识别瞌睡的系统，那么无疑该系统意义重大！
**教程博客_传送门链接------->[疲劳检测](https://blog.csdn.net/ALiLiLiYa/article/details/132515440)**
![在这里插入图片描述](https://img-blog.csdnimg.cn/e5e9c9c1aad34745a8543fda5a0e8d01.png)


## 车辆跟踪及测距

该项目一个基于深度学习和目标跟踪算法的项目，主要用于实现视频中的目标检测和跟踪。该项目使用了 YOLOv4 目标检测算法和 DeepSORT 目标跟踪算法，以及一些辅助工具和库，可以帮助用户快速地在本地或者云端上实现视频目标检测和跟踪！

**教程博客_传送门链接------->[单目测距和跟踪](https://blog.csdn.net/ALiLiLiYa/article/details/129822610)**

![在这里插入图片描述](https://img-blog.csdnimg.cn/a7a09fcd8a4b483a9b76c5550a37e4f5.png)

# yolov5 deepsort 行人/车辆（检测 +计数+跟踪+测距+测速）

 - 实现了局域的出/入 分别计数。
 - 显示检测类别，ID数量。
 - 默认是 南/北 方向检测，若要检测不同位置和方向，需要加以修改
 - 可在 count_car/traffic.py 点击运行
 - 默认检测类别：行人、自行车、小汽车、摩托车、公交车、卡车、船。
 - 检测类别可在 objdetector.py 文件修改。

**原文链接：https://blog.csdn.net/ALiLiLiYa/article/details/131819630**

![在这里插入图片描述](https://img-blog.csdnimg.cn/62d1f83f6de946daa556267c42cc3ff3.png)

# 目标跟踪

**教程博客_传送门链接------->[目标跟踪](https://blog.csdn.net/ALiLiLiYa/article/details/131741399)**

YOLOv5是一种流行的目标检测算法，它是YOLO系列算法的最新版本。YOLOv5采用了一种新的架构，可以在保持高准确性的同时提高检测速度。在本文中，我们将介绍如何使用YOLOv5_deepsort算法来进行船舶跟踪和测距。

![在这里插入图片描述](https://img-blog.csdnimg.cn/84f139a34cbf456ca6a01f44d32e0481.png)

# 车道线识别

本文主要讲述项目集成：从车道线识别、测距、到追踪，集各种流行模型于一体！

不讲原理，直接上干货！

把下文环境配置学会，受益终生！

各大项目皆适用！


**教程博客_传送门链接------->[车道线识别+目标检测](https://blog.csdn.net/ALiLiLiYa/article/details/131610493)**
看下本项目的效果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/1942012c55f6454d81da1f3b9d4c482c.png)

# 语义分割

MMsegmentation是一个基于PyTorch的图像分割工具库，它提供了多种分割算法的实现，包括语义分割、实例分割、轮廓分割等。MMsegmentation的目标是提供一个易于使用、高效、灵活且可扩展的平台，以便开发者可以轻松地使用最先进的分割算法进行研究和开发。
**教程博客_传送门链接------->[语义分割](https://blog.csdn.net/ALiLiLiYa/article/details/130836710)**

![在这里插入图片描述](https://img-blog.csdnimg.cn/3073c1c0947245adb56b0fa2461f8c60.png)

# 姿态识别

人体姿态估计是计算机视觉中的一项重要任务，具有各种应用，例如动作识别、人机交互和监控。近年来，基于深度学习的方法在人体姿态估计方面取得了显著的性能。其中最流行的深度学习方法之一是YOLOv7姿态估计模型。
**程博客_传送门链接------->：https://blog.csdn.net/ALiLiLiYa/article/details/129482358**

![在这里插入图片描述](https://img-blog.csdnimg.cn/6a1425075b5d4467af5e69b03b3472cf.png)
