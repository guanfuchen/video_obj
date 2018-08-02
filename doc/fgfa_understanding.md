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

如下是dff_rfcn和rfcn的检测对比，发现虽然速度比rfcn单帧检测快2X左右，但是dff_rfcn（论文提出的方法）检测精度远远低于rfcn单帧检测，具体如下表所示，由于demo中没有标记数据，因此检测精度未给出量化指标，另外给出论文在VID校准集上的精度mAP。

|  | time/images | 检测精度 | mAP@0.5 |
| ---- | ---- | ---- | ---- |
| rfcn | **0.133s** | 低 | 73.0 |
| dff_rfcn | 0.034s | **中** | **74.1** |
| fgfa | 0.800s | **高** | **83.5** |


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

TODO增加光流网络相关细节。

---
## 参考资料

- [Flow-Guided-Feature-Aggregation](https://github.com/msracver/Flow-Guided-Feature-Aggregation)。
