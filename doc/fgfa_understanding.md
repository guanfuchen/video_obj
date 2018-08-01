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

#### 下一步工作

阅读相关代码，结合论文解析代码。

#### 编译问题

- ...

---
## 参考资料

- [Flow-Guided-Feature-Aggregation](https://github.com/msracver/Flow-Guided-Feature-Aggregation)。
