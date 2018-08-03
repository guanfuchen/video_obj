# FGFA

Flow-guided feature aggregation for video object detection

---
## 相关描述

使用了特征增强的方法来提升视频目标检测的精度。

---
## 源码安装

### 安装MXNet

同dff_understanding文件描述。

### FGFA使用

```
运行demo前启动下述初始化脚本
sh ./init.sh

运行demo.py
CUDA_VISIBLE_DEVICES=1 python ./fgfa_rfcn/demo.py

发现错误Symbol can not use “+=”，参考[Symbol can not use “+=”](https://github.com/msracver/Flow-Guided-Feature-Aggregation/issues/13)，
修改
aggregated_conv_feat += tiled_weight * warp_list[i]
为
aggregated_conv_feat = aggregated_conv_feat + tiled_weight * warp_list[i]
```

下载预训练模型。

#### 检测效果

> 如下是dff_rfcn和rfcn的检测对比，发现虽然速度比rfcn单帧检测快2X左右，但是dff_rfcn（论文提出的方法）检测精度远远低于rfcn单帧检测，具体如下表所示，由于demo中没有标记数据，因此检测精度未给出量化指标，另外给出论文在VID校准集上的精度mAP。

本文的FGFA架构则在检测精度上大大提升。

|  | time/images | 检测精度 | mAP@0.5 |
| ---- | ---- | ---- | ---- |
| rfcn | **0.133s** | 低 | 73.0 |
| dff_rfcn | 0.034s | **中** | **74.1** |
| fgfa | 0.800s | **高** | **83.5** |

#### 不同的motion IOU检测结果

将运动分为slow，medium，fast运动，不同的motion下分别检测mAP，来体现FGFA模型在fast运动目标检测任务上的精度提升较大。运动分类通过和邻近帧（前后10帧）相应目标实例的平均IOU标准分为slow，medium和fast运动，这里将这个评价标准定为motion IOU，motion IOU更低也就是目标运动更快，具体分类指标如下表所示。

|  | motion IOU |
| ---- | ---- |
| slow | >0.9 |
| medium | [0.7, 0.9] |
| fast | <0.7 |

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/fgfa_motion_iou.png)

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/fgfa_motion_iou_ex.png)

```
===========================================
eval_vid_detection :: accumulating: motion [0.0 1.0], area [0.0 0.0 100000.0 100000.0]
===========================================
eval_vid_detection :: accumulating: motion [0.0 0.7], area [0.0 0.0 100000.0 100000.0]
===========================================
eval_vid_detection :: accumulating: motion [0.7 0.9], area [0.0 0.0 100000.0 100000.0]
===========================================
eval_vid_detection :: accumulating: motion [0.9 1.0], area [0.0 0.0 100000.0 100000.0]
=================================================
motion [0.0 1.0], area [0.0 0.0 100000.0 100000.0]
Mean AP@0.5 = 0.7711
=================================================
motion [0.0 0.7], area [0.0 0.0 100000.0 100000.0]
Mean AP@0.5 = 0.5611
=================================================
motion [0.7 0.9], area [0.0 0.0 100000.0 100000.0]
Mean AP@0.5 = 0.7567
=================================================
motion [0.9 1.0], area [0.0 0.0 100000.0 100000.0]
Mean AP@0.5 = 0.8591
```



#### 训练

该训练需要下载**ILSVRC2015 DET**和**ILSVRC2015 VID**数据集。

#### 下一步工作

阅读相关代码，结合论文解析代码。

#### 编译问题

- ...

### 代码阅读

#### 特征网络Feature Network

这里使用ResNet模型，代码给了ResNet-101模型示例，修改如下：
- 去除最后的1000维度的分类网络；
- 将特征图从stride为32修改为stride为16；
- conv5的第一个网络层将stride从2修改为1；
- conv5中的3x3卷积核应用holing算法保持感受野（dilation=2）；
- conv5中最后的3x3卷积核随机初始化，并且修改特征通道维度为1024，并且使用了holing算法（dilation=6）。

具体代码修改如下：

```python

# conv5的第一个网络层将stride从2修改为1，其余都正常为2
res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu, num_filter=512, pad=(0, 0), kernel=(1, 1), stride=(2, 2), no_bias=True)

# conv5的第一个网络层将stride从2修改为1；
res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=res4b22_relu, num_filter=2048, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)

# conv5中的3x3卷积核应用holing算法保持感受野（dilation=2）
res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu, num_filter=512, pad=(2, 2), dilate=(2, 2), kernel=(3, 3), stride=(1, 1), no_bias=True)
res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu, num_filter=512, pad=(2, 2), dilate=(2, 2), kernel=(3, 3), stride=(1, 1), no_bias=True)
res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu, num_filter=512, pad=(2, 2), dilate=(2, 2), kernel=(3, 3), stride=(1, 1), no_bias=True)

# conv5中最后的3x3卷积核随机初始化，并且修改特征通道维度为1024，并且使用了holing算法（dilation=6）
feat_conv_3x3 = mx.sym.Convolution(data=res5c_relu, kernel=(3, 3), pad=(6, 6), dilate=(6, 6), num_filter=1024, name="feat_conv_3x3")
```

#### Emdedding Network

Emdedding Network输入为特征网络的输出，由3个网络层构造而成，包括1x1x512,3x3x512和1x1x2048的卷积。
该网络主要用来将特征网络投射到一个新的embedding作为相似性测量（这里相似性测量使用cosine相似性测量标准），这样就可以将邻近帧的特征在新的embedding中测量相似性，相似性较大的邻近帧特征聚合权值较大，相似性较小的邻近帧特征聚合权值较小，其中$f^e=\epsilon(f)$表示使用Embedding Network计算对应的Embedding作为相似性度量的输入，具体计算如下所示：

$$w_{j \rightarrow i}(p)=exp(\frac{f_{j \rightarrow i}^{e}(p) \cdot f_{i}^{e}(p)}{|f_{j \rightarrow i}^{e}(p)| \cdot |f_{i}^{e}(p)|})$$

$$f^e=\epsilon(f)$$

最后需要将自适应权重值归一化$\sum_{j=i-K}^{i+K} w_{j \rightarrow i}(p)=1$。
以上这些过程可以理解为首先L2 Norm归一化$f_{j \rightarrow i}^{e}(p)$和$f_{i}^{e}(p)$，然后相乘，最后使用softmax函数$y_i=\frac{e^{x_i}}{\sum_{j}e^{x_j}}$即可得到最后的归一化的自适应权重，这个思路也是代码中应用的思路。

### 光流网络Flow Network

TODO增加光流网络相关细节，具体内容查看[flownet_understanding.md](./flownet_understanding.md)。

### 检测网络Detection Network

本文使用了state-of-the-art R-FCN目标检测网络，在1024维度的特征map后紧跟着RPN子网络和R-FCN子网络，RPN网络中使用了9个锚点（3种不同的尺度和3种不同的aspect ratios），每一张图像产生了300个区域建议。R-FCN中位置敏感的score maps分为7x7的groups。


### 训练细节

训练同时在ImageNet Det和ImageNet VID数据集上训练，分为两阶段训练。
- 第一阶段在ImageNet DET数据集上训练特征网络Feature Network和检测网络Detection Network（使用ImageNet VID标注DET标注的30类目标），优化方法采用SGD，每一个mini-batch中使用一张图像进行训练，共使用4张显卡，训练周期为120K，前80K次迭代学习率为$10^{-3}$，后40K次迭代学习率为$10^{-4}$。
- 第二阶段在ImageNet VID上训练整个FGFA模型，其中特征网络Feature Network和检测网络Detection Network从第一阶段学习到的权重初始化得到，在4张显卡上执行60K次迭代，前40K次迭代学习率为$10^{-3}$，后20K次迭代学习率为$10^{-4}$。训练和测试期间，输入到特征网络Feature Network中的较短边为600像素，输入到光流网络Flow Network中的较短边为300像素。

### 实验相关

#### Box-level技术组合

该实验对于常用的video object detection的Box-level技术组合到FPFA架构中，比如MGP，Tubelet rescoring和Seq-NMS，实验发现这些方法对于ImageNet VID单帧检测的结果都有较大的提升，其中Seq-NMS方法提升最大，但是和FPFA架构组合后，MGP和Tubelet rescoring这两种方法几乎没有什么精度的提升，但是Seq-NMS仍然给了较大的提升结果，通过使用Aligned-Inception-ResNet作为特征网络的方法提升到了80.1%。

---
## 参考资料

- [Flow-Guided-Feature-Aggregation](https://github.com/msracver/Flow-Guided-Feature-Aggregation)。
- [proposal-inl.h](https://github.com/apache/incubator-mxnet/blob/master/src/operator/contrib/proposal-inl.h) MXNet中相应的Proposal网络层源码。
- [Deformable-ConvNets](https://github.com/bharatsingh430/Deformable-ConvNets) 第三方的实现D-R-FCN+Soft-NMS。
