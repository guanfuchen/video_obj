# video_obj

基于视频的目标检测算法研究

对相应的视频目标检测论文整理实现综述文档。

---
## 数据集

### ILSVRC2015: Object detection from video (VID)



### YouTube-Objects dataset v2.2

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/Screen_Shot_2018-07-11_16.35.20.png)

YouTube-Objects数据集由从YouTube收集的视频组成，查询PASCAL VOC Challenge的10个对象类别的名称。每个对象包含9到24个视频。每个视频的持续时间在30秒到3分钟之间变化。视频被弱标注，即我们确保每个视频包含相应类的至少一个对象。该数据集包括aeroplane、bird、boat、car、cat、cow、dog、horse、motorbike和train这10个类别，具体可在网页上查看[YouTube-Objects v2.3 Preview](YouTube-Objects v2.3 Preview)。

[YouTube-Objects dataset v2.3](http://calvin.inf.ed.ac.uk/datasets/youtube-objects-dataset/) yto目标检测数据集主页。

[yto-dataset](https://github.com/vkalogeiton/yto-dataset) yto数据集下载和使用说明。

- Learning Object Class Detectors from Weakly Annotated Video
- Analysing domain shift factors between videos and images for object detection

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


---
## Object Detection from Video Tubelets with Convolutional Neural Networks

目标定位和联合定位和VID任务似乎有着相似的topic，但是这两个问题有着本质的区别。（1）目标：目标定位或者联合定位问题假设每一个视频仅仅包含一个已知或者未知的类别，并且仅仅要求定位下一帧目标的一个物体。在VID任务中，每一个视频帧包含了未知数量的实例或者类别。VID任务更接近与真实应用。（2）评估指标：定位的评估指标通常被用来评估定位的精度，也就是在VID任务中使用的mAP。

本文主要使用了时空tubelet建议模块组合了静止图像的目标检测和通用的目标跟踪。因此该模块同时具有目标检测器的识别能力和目标跟踪器的时间一致性能力。该模块主要有三步：（1）图像目标建议，（2）目标建议打分和（3）高置信度目标跟踪。

---
## Optimizing Video Object Detection via a Scale-Time Lattice

本文探索了使用一种新的方法，在规模时间内重新分配计算空间。

具体来说，在自然视频中的帧中存在很强的连续性，这表明了另一种可选的减少计算成本的方法，即时序上传播计算。

通常来说，基于视频的目标检测方法是一个多步骤的过程，先前研究的任务中，比如基于图像的目标检测，时序传播，稀疏到细致化的微调等等都是这个过程中的单一步骤。然而单一步骤的提升尽管被研究了很久，但是一个关键问题仍然悬而未决：“什么是最具成本效益地将它们结合起来的策略？”

Scale-Time Lattice是一个统一的形式，其中上面提到的步骤是Scale-Time Lattice中有向连接的不同节点。 从这个统一的观点来看，可以很容易看出不同的步骤如何贡献以及如何计算成本分配。

---
## Deep Feature Flow for Video Recognition

现代的CNN网络架构共享相同的结构。大部分网络层是卷积并且因此导致了最大的计算代价。中间的卷积特征map和输入图像有着相似的空间extent（通常更小的分辨率，比如小16X）。它们在low level的图像内容和中高级语义概念保持了空间的对应性。这种对应性能够提供使用空间warping（和光流法相似）将邻近帧的特征轻量传播的用处。

在这项工作中，我们提出了深入的特征流，快速和准确的视频识别方法。 它应用了一个图像稀疏关键帧上的识别网络。 它传播深度特征从关键帧映射到其他帧流场。 如图1中所示，两个中间体特征地图响应“汽车”和“人”概念。它们在附近的两个框架上相似。 传播后，传播的特征与原始特征类似。

通常，光流估计和特征传播比卷积特征的计算快得多。因此，避免了计算瓶颈实现了显着的加速。 当流场也是通过网络估计，整个架构都经过培训端到端，具有图像识别和流网络针对识别任务进行了优化。 识别准确性显着提升。

**简要可以这么理解，在关键帧使用稠密的网络进行检测，同时保存保留了一致性的先前的网络特征，非关键帧使用关键帧的这个网络特征以及稀疏的网络（预测光流）进行检测。**

总结来说，深度特征流方法DFF是一个用来视频识别的快速精确，通用的端到端的框架。

本文提出的方法示意图如下所示，其中第一列为关键帧的原图，网络结构183和289输出的卷积特征，第二列为当前帧的原图，网络结构183和289输出的卷积特征，第三列为当前帧的光流估计和通过计算的传播的特征map，可以看出通过使用关键帧的卷及特征和光流的传播的特征map和当前帧直接在网络的输出几乎相同。

![](D:/GitHub/Quick/实习工作/imgs/dff_result.png)

本文提出的网络处理过程和每一帧的网络框架区别如下所示，其中每一帧网络per-frame network处理每一帧，并且每一帧都会输入特征提取网络提取特征，同时将提取的特征输入到识别任务中输出最后的任务结果，而本文提出的DFF深度特征光流网络DFF网络仅仅对关键帧提取特征，然后当前帧（非关键帧，即两个关键帧之间的frame）和关键帧输入到光流估计函数F中，将关键帧提取的特征和光流估计结果输入至传播函数propagation中，然后输入到输出task任务中得到当前帧的任务结果。

![](D:/GitHub/Quick/实习工作/imgs/dff_illustration.png)


### 参考资料

- [读书笔记Deep Feature Flow for Video Recognition](https://zhuanlan.zhihu.com/p/27213979)
- [视频检测分割--Deep Feature Flow for Video Recognition](https://blog.csdn.net/zhangjunhit/article/details/76665253)
- [视频物体检测文献阅读笔记](https://blog.csdn.net/Wayne2019/article/details/78927733)

---
## Object detection in videos with tubelet proposal networks

参考代码[TPN](https://github.com/myfavouritekk/TPN) 相较于RPN，生成了一系列基于视频管道的区域建议。

---
## FlowNet: Learning Optical Flow with Convolutional Networks

使用卷积网络学习光流估计。

### 参考资料

- [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)实现了flownet2。
- [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch)参考该实现，对FlowNet网络结构进行相应的了解。


