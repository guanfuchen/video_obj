# video_obj

基于视频的目标检测算法研究

对相应的视频目标检测论文整理实现综述文档。

---
## 数据集

### ILSVRC2015: Object detection from video (VID)
ImageNet VID challenges，这是在kaggle上的关于ImageNet上基于视频的目标检测挑战，目的是为了识别和标记视频中的普通目标。

该数据集文件如下
- imagenet_object_detection_video_train.tar.gz包含了训练集和校准集的图像数据和GT。
- imagenet_object_detection_video_test.tar.gz包含了测试机的图像数据。
  - 其中图像标注格式都是基于PASCAL VOC数据集格式的XML文件（可以使用PASCAL开发工具套件来解析标注）。
  - 每一个视频都是以JPEG格式存储，代表不同帧。
  - ImageSet文件夹包含了定义了主要的检测任务的图像列表。例如，文件夹ILSVRC2015_VID_train_0000/ILSVRC2015_train_00025030表示一个视频，其中该文件夹中的000000.JPEG文件表示第一帧，并且000000.xml表示该帧的标注。


### YouTube-Objects dataset v2.2

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/Screen_Shot_2018-07-11_16.35.20.png)

YouTube-Objects数据集由从YouTube收集的视频组成，查询PASCAL VOC Challenge的10个对象类别的名称。每个对象包含9到24个视频。每个视频的持续时间在30秒到3分钟之间变化。视频被弱标注，即我们确保每个视频包含相应类的至少一个对象。该数据集包括aeroplane、bird、boat、car、cat、cow、dog、horse、motorbike和train这10个类别，具体可在网页上查看[YouTube-Objects v2.3 Preview](YouTube-Objects v2.3 Preview)。

[YouTube-Objects dataset v2.3](http://calvin.inf.ed.ac.uk/datasets/youtube-objects-dataset/) yto目标检测数据集主页。

[yto-dataset](https://github.com/vkalogeiton/yto-dataset) yto数据集下载和使用说明。

- Learning Object Class Detectors from Weakly Annotated Video
- Analysing domain shift factors between videos and images for object detection

### YouTube-BoundingBoxes: A Large High-Precision Human-Annotated Data Set for Object Detection in Video

该数据集中包含单个目标。

---
## 相关资料

- [ImageNet Object Detection from Video Challenge](https://www.kaggle.com/c/imagenet-object-detection-from-video-challenge) kaggle上的一个ImageNet基于视频的目标检测比赛，可以作为初始数据集测试相应的算法。
- [Optimizing Video Object Detection via a Scale-Time Lattice](https://arxiv.org/pdf/1804.05472.pdf) 推荐阅读的一篇相关论文。
- [FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852) 这篇文章介绍了使用CNN来计算光流的模型。
- [Video Object Detection](https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-object-detection.md#video-object-detection) handong1587对视频目标检测相关论文的收集。
- Learning Object Class Detectors from Weakly Annotated Video
- Analysing domain shift factors between videos and images for object detection
- T-CNN: Tubelets with Convolutional Neural Networks for Object Detection from Videos
- Object Detection from Video Tubelets with Convolutional Neural Networks
- Object Detection in Videos with Tubelets and Multi-context Cues
- Context Matters: Refining Object Detection in Video with Recurrent Neural Networks
- Object Detection in Videos with Tubelet Proposal Networks
- CNN Based Object Detection in Large Video Images幻灯片
- Flow-Guided Feature Aggregation for Video Object Detection
- Object Detection in Video using Faster R-CNN
- Impression Network for Video Object Detection
- Towards High Performance Video Object Detection for Mobiles
- Temporal Dynamic Graph LSTM for Action-driven Video Object Detection
- Mobile Video Object Detection with Temporally-Aware Feature Maps
- Towards High Performance Video Object Detection
- Object Detection with an Aligned Spatial-Temporal Memory
- 3D-DETNet: a Single Stage Video-Based Vehicle Detector
- Improving Context Modeling for Video Object Detection and Tracking VID挑战PPT。
- Semantic Video CNNs through Representation Warping
- Clockwork Convnets for Video Semantic Segmentation
- Slow Feature Analysis_ Unsupervised Learning of Invariances慢特征分析，主要基于连续的视频关键帧特征具有极大的相似性这个特点提取信息。
- Deep Learning of Invariant Features via Simulated Fixations in Video
- Slow and steady feature analysis: higher order temporal coherence in video
- Seq-NMS for Video Object Detection将传统的基于still image的区域建议NMS方法扩展到视频序列的NMS方法。
- The Recognition of Human Movement Using Temporal Templates，论文提出了Motion History Image（MHI）作为运动表示，该表示计算高效，对于基于光流的方法来说可以作为其替代来弥补光流计算量大的问题 TODO。
- Detect to Track and Track to Detect


---
## YouTube-BoundingBoxes: A Large High-Precision Human-Annotated Data Set for Object Detection in Video

YouTube-BoundingBoxes：用于视频中对象检测的大型高精度人体注释数据集，下载地址[youtube-bb](https://research.google.com/youtube-bb/)，浏览地址[BoundingBoxes](https://research.google.com/youtube-bb/explore.html)。该数据集包含大约38,000个约19秒长的视频片段，自动选择自然设置中的特征对象而无需编辑或后处理，其录制质量通常类似于手持式手机相机。

本文中的相关工作介绍了视频目标检测领域的数据集和静态图像的目标检测领域的数据集，包括VOT、MOT等等。

数据集预览界面如下所示：

![](./imgs/ytbb_vis.png)

数据集包含如下四个CSV文件:
- 视屏segments中的分类 - 训练集 (27Mb gzip压缩文件)
- 视屏segments中的分类 - 校准集 (3.4Mb gzip压缩文件)
- 视屏segments中的检测 - 训练集 (57Mb gzip压缩文件)
- 视屏segments中的检测 - 校准集 (6.3Mb gzip压缩文件)

在检测CSV文件中，每一行表示一帧并且每一列如下所示：
- youtube_id 分割被提取的视屏的YouTube分类号，组合网址http://youtube/%{youtube_id}跟踪到选择的视屏
- timestamp_ms 视频中检测帧的时间ms
- class_id 目标类别的数值标注
- class_name 人类可读的目标类别名
- object_presence 目标是否在当前帧中
- xmin [0.0, 1.0]boundingx box最左边相对于帧大小的位置
- xmax [0.0, 1.0]boundingx box最右边相对于帧大小的位置
- ymin [0.0, 1.0]boundingx box最上边相对于帧大小的位置
- ymax [0.0, 1.0]boundingx box最下边相对于帧大小的位置

如下所示：

```
AAB6lO-XiKE	238000	0	person	0	present	0.482	0.54	0.37166667	0.6166667
AAB6lO-XiKE	239000	0	person	0	present	0.514	0.588	0.36333334	0.6066667
AAB6lO-XiKE	240000	0	person	0	present	0.534	0.614	0.44333333	0.685
AAB6lO-XiKE	241000	0	person	0	present	0.515	0.605	0.44833332	0.68666667
```


每一个视频分割片段中最多只有一个目标被跟踪，但是同一个视频中能够有多个分割，也就是说youtube_id可能有多个分割，但是youtube_id和class_id组合就只有唯一的跟踪。

---
## Object Detection from Video Tubelets with Convolutional Neural Networks

目标定位和联合定位和VID任务似乎有着相似的topic，但是这两个问题有着本质的区别。（1）目标：目标定位或者联合定位问题假设每一个视频仅仅包含一个已知或者未知的类别，并且仅仅要求定位下一帧目标的一个物体。在VID任务中，每一个视频帧包含了未知数量的实例或者类别。VID任务更接近与真实应用。（2）评估指标：定位的评估指标通常被用来评估定位的精度，也就是在VID任务中使用的mAP。

本文主要使用了时空tubelet建议模块组合了静止图像的目标检测和通用的目标跟踪。因此该模块同时具有目标检测器的识别能力和目标跟踪器的时间一致性能力。该模块主要有三步：（1）图像目标建议，（2）目标建议打分和（3）高置信度目标跟踪。


---
## T-CNN: Tubelets with Convolutional Neural Networks for Object Detection from Videos

使用两个多阶段更快的R-CNN [31]检测框架，上下文抑制，多尺度训练/测试，ConvNet跟踪器[39]，基于光流的分数传播 和模特合奏。

---
## Object detection in videos with tubelet proposal networks

...

---
## Optimizing Video Object Detection via a Scale-Time Lattice

本文探索了使用一种新的方法，在规模时间内重新分配计算空间。

具体来说，在自然视频中的帧中存在很强的连续性，这表明了另一种可选的减少计算成本的方法，即时序上传播计算。

通常来说，基于视频的目标检测方法是一个多步骤的过程，先前研究的任务中，比如基于图像的目标检测，时序传播，稀疏到细致化的微调等等都是这个过程中的单一步骤。然而单一步骤的提升尽管被研究了很久，但是一个关键问题仍然悬而未决：“什么是最具成本效益地将它们结合起来的策略？”

Scale-Time Lattice是一个统一的形式，其中上面提到的步骤是Scale-Time Lattice中有向连接的不同节点。 从这个统一的观点来看，可以很容易看出不同的步骤如何贡献以及如何计算成本分配。

文中实验结果比较了常用的在VID数据集上实验的方法，其中包括DFF、TPN+LSTM、FGFA和D&T，以及本文提出的scale-time lattice方法，具体比较结果如下图所示：

![](./imgs/vid_dataset_solution_results.png)

另外不同于DFF使用光流来传播关键帧的稠密特征，本文主要使用MHI来编码运动信息来传播帧间运动特征，下图比较了在不同间隔的关键帧下的不同传播方法的精度，左图是整体精度比较，右图是基于不同的目标运动的检测精度比较，其中比较主要包括Interpolation、RGB差值和MHI这三种方法，另外从右图中可以看出使用MHI方法精度提升的主要目标位快速运动的目标。

![](./imgs/propagation_result.png)

---
## Detect to Track and Track to Detect

相关代码[Detect-Track](https://github.com/feichtenhofer/Detect-Track)和[py-Detect-Track代码python](https://github.com/feichtenhofer/py-Detect-Track)

文章指出：在视频中的对象检测和跟踪的情况下，最近的方法主要使用检测作为第一步，接着是后处理方法，诸如应用跟踪器以随时间传播检测分数。 “检测跟踪”范式的这种变化已经取得了令人瞩目的进展，但却受到帧级检测方法的支配。

视频中的对象检测最近引起了人们的兴趣，尤其是在引入ImageNet视频对象检测挑战（VID）之后。 与ImageNet对象检测（DET）挑战不同，VID在图像序列中显示对象，并带来额外的挑战：（i）大小：视频提供的帧数（VID大约有130万图像，而DET大约有400K） COCO大约有100K），（ii）运动模糊：由于快速的相机或物体运动，（iii）质量：互联网视频剪辑的质量通常低于静态照片，（iv）部分遮挡：由于 物体/观察者定位，以及（v）姿势：在视频中经常看到非常规的物体到相机姿势。 在下图中，我们显示了来自VID数据集的示例图像。

![](./imgs/vid_samples.png)

---
## Deep Feature Flow for Video Recognition

现代的CNN网络架构共享相同的结构。大部分网络层是卷积并且因此导致了最大的计算代价。中间的卷积特征map和输入图像有着相似的空间extent（通常更小的分辨率，比如小16X）。它们在low level的图像内容和中高级语义概念保持了空间的对应性。这种对应性能够提供使用空间warping（和光流法相似）将邻近帧的特征轻量传播的用处。

在这项工作中，我们提出了深入的特征流，快速和准确的视频识别方法。 它应用了一个图像稀疏关键帧上的识别网络。 它传播深度特征从关键帧映射到其他帧流场。 如图1中所示，两个中间体特征地图响应“汽车”和“人”概念。它们在附近的两个框架上相似。 传播后，传播的特征与原始特征类似。

通常，光流估计和特征传播比卷积特征的计算快得多。因此，避免了计算瓶颈实现了显着的加速。 当流场也是通过网络估计，整个架构都经过培训端到端，具有图像识别和流网络针对识别任务进行了优化。 识别准确性显着提升。

**简要可以这么理解，在关键帧使用稠密的网络进行检测，同时保存保留了一致性的先前的网络特征，非关键帧使用关键帧的这个网络特征以及稀疏的网络（预测光流）进行检测。**

总结来说，深度特征流方法DFF是一个用来视频识别的快速精确，通用的端到端的框架。

本文提出的方法示意图如下所示，其中第一列为关键帧的原图，网络结构183和289输出的卷积特征，第二列为当前帧的原图，网络结构183和289输出的卷积特征，第三列为当前帧的光流估计和通过计算的传播的特征map，可以看出通过使用关键帧的卷及特征和光流的传播的特征map和当前帧直接在网络的输出几乎相同。

![](./imgs/dff_result.png)

本文提出的网络处理过程和每一帧的网络框架区别如下所示，其中每一帧网络per-frame network处理每一帧，并且每一帧都会输入特征提取网络提取特征，同时将提取的特征输入到识别任务中输出最后的任务结果，而本文提出的DFF深度特征光流网络DFF网络仅仅对关键帧提取特征，然后当前帧（非关键帧，即两个关键帧之间的frame）和关键帧输入到光流估计函数F中，将关键帧提取的特征和光流估计结果输入至传播函数propagation中，然后输入到输出task任务中得到当前帧的任务结果。

![](./imgs/dff_illustration.png)


### 参考资料

- [读书笔记Deep Feature Flow for Video Recognition](https://zhuanlan.zhihu.com/p/27213979)
- [视频检测分割--Deep Feature Flow for Video Recognition](https://blog.csdn.net/zhangjunhit/article/details/76665253)
- [视频物体检测文献阅读笔记](https://blog.csdn.net/Wayne2019/article/details/78927733)

---
## Object detection in videos with tubelet proposal networks

参考代码[TPN](https://github.com/myfavouritekk/TPN) 相较于RPN，生成了一系列基于视频管道的区域建议。

---
## Optimizing Video Object Detection via a Scale-Time Lattice

- [scale-time-lattice相关代码](https://github.com/hellock/scale-time-lattice)

网络结构如下所示，其中小红点表示在关键帧的检测，方格点表示尺度-时间格子，也就是空间-时间格子的结果，其中黑色虚线表示直接映射或者缩放，蓝色实线表示在空间上的传播，蓝色实线表示在空间上的微调，图中水平方向的操作是在时间上的传播，垂直方向的操作是在空间上的细化，其中PRU表示Propagation and Refinement Unit，即传播细化单元，这个基本结构是构成格子的主要组件，用来完成时间传播和空间细化：

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/Screen_Shot_2018-07-12_23.08.31.png)

PRU将两个连续的关键帧的检测结果作为输入，然后传播到参考帧中国年，并且通过细化输出到下一空间尺度。

### 参考资料

- [Optimizing Video Object Detection via a Scale-Time Lattice](https://amds123.github.io/2018/04/16/Optimizing-Video-Object-Detection-via-a-Scale-Time-Lattice/) 中文摘要。

---
## Flow-guided feature aggregation for video object detection

和deep feature flow的思路相似，通过光流的方法增强视频目标检测。


# flow

增加和video_obj并行的光流论文研究。

---
## FlowNet: Learning Optical Flow with Convolutional Networks

使用卷积网络学习光流估计。

### 参考资料

- [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)实现了flownet2。
- [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch)参考该实现，对FlowNet网络结构进行相应的了解。
- [论文笔记：FlowNet](https://calmdownandcarryon.github.io/2017/09/08/paper-reading/iccv_flownet/)
- [CNN光流计算--FlowNet: Learning Optical Flow with Convolutional Networks](https://blog.csdn.net/zhangjunhit/article/details/76262429)

---
## Unsupervised Learning of Depth and Ego-Motion from Video

[Unsupervised Learning of Depth and Ego-Motion from Videox项目主页](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/)

---
# GPU

NVIDIA® Tesla® P100 GPU 加速器为现代数据中心释放强大的计算能力。它利用全新的 NVIDIA Pascal™ 架构打造出速度极快的计算节点，性能高于数百个速度较慢的通用计算节点。利用更少的快速的节点获得更高的性能，能在节省资金的同时，大幅提高数据中心吞吐量。

## 参考资料

- [NVIDIA® TESLA® P100](https://www.nvidia.cn/object/tesla-p100-cn.html)