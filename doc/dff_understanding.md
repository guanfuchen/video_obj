# DFF

Deep Feature Flow for Video Recognition

---
## 相关描述

一般认为图像内容在视频上的变化是缓慢的，尤其是高级语义。因此在特征学习时采用means of regularization，认为视频是一种无监督的数据源。并且，数据冗余和连续性也可以用于减少计算开销。但是相关的研究在视频识别的CNN上却较少。

现代CNN具有相同的架构，大部分层为卷积计算，卷积特征与原图具有空间对应性，因此有机会通过空间扭曲将特征传播到邻近帧。

---
## 源码安装

### 安装MXNet
```
apt-get update
apt-get install -y build-essential git
apt-get install -y libopenblas-dev
apt-get install -y libopencv-dev
make -j4 USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1

从最初的cuda8_cudnn6的镜像建立的docker进行源码编译失败，然后找到了预先编译的mxnet镜像，准备加载镜像尝试编译成功。（该镜像重新编译了git，所以可以通过使用git clone！！！成功下载git仓库！！！）
cp -r $(DFF_ROOT)/dff_rfcn/operator_cxx/* $(MXNET_ROOT)/src/operator/contrib/
make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 USE_PROFILER=1
cd python
python setup.py install
```

### DFF使用

```
运行demo前启动下述初始化脚本
sh ./init.sh
编译过程中出现lib/nms文件下的相关gpu编译错误，看错误信息较难定位bug，此时联想到小卫老师给的face model代码库中nms可以编译成功，对比文件发现基本一样，所以将face model代码库中的nms替换相应的文件即可，编译成功。

运行demo.py
CUDA_VISIBLE_DEVICES=1 python ./rfcn/demo.py

发现opencv的一些错误，需要安装opencv3版本，但是由于mxnet镜像中用到的是其他人的http_proxy（密码已经失效），pip修改代理依旧不行，但是在conda修改代理可以，需要应用conda install opencv安装依赖。

发现错误Request: finish python gpu enabled guide for install，参考[Request: finish python gpu enabled guide for install](https://github.com/apache/incubator-mxnet/issues/7900)，发现应该make clean整个MXNet模型，然后重新make一下，最后将python目录make install即可，相同问题可以参考[FCIS demo - error](https://github.com/msracver/FCIS/issues/10)。
```

下载预训练模型，其中包含pretrained_demo.zip（谭博帮忙下载）和demo_model.zip（自己下载拷贝到电脑上传输到服务器上，使用小卫老师的账号密码10.177.130.174 w00416240 board_143），最后上传到自己的云桌面中然后拷贝到10.154.67.148的服务器中。

#### 检测效果

如下是dff_rfcn和rfcn的检测对比，发现虽然速度比rfcn单帧检测快2X左右，但是dff_rfcn（论文提出的方法）检测精度远远低于rfcn单帧检测，具体如下表所示，由于demo中没有标记数据，因此检测精度未给出量化指标，另外给出论文在VID校准集上的精度mAP。

|  | time/images | 检测精度 | mAP@0.5 |
| ---- | ---- | ---- | ---- |
| rfcn | **0.133s** | 低 | 73.0 |
| dff_rfcn | 0.034s | **高** | **74.1** |

下图是dff_rfcn和rfcn的检测比较：


![30%](imgs/rfcn_dff/000000.JPEG)

![30%](imgs/rfcn/000000.JPEG)

<!--
![30%](imgs/rfcn_dff/000001.JPEG)
![30%](imgs/rfcn/000001.JPEG)
![30%](imgs/rfcn_dff/000002.JPEG)
![30%](imgs/rfcn/000002.JPEG)
-->

#### 下一步工作

由于DFF是该系列论文的第一篇，其余针对精度（Towards High Performance Video Object Detection）和速度（Towards High Performance Video Object Detection for Mobiles）都做了相关研究，这里尝试这些能否提升检测效果。

#### 编译问题

- [ImportError: cannot import name 'bbox_overlaps_cython'](https://github.com/msracver/Deep-Feature-Flow/issues/21)

---
## 参考资料

- [Deep Feature Flow for Video Recognition](https://github.com/msracver/Deep-Feature-Flow)。
- [MXNet - Ubuntu安装](https://blog.csdn.net/zziahgf/article/details/72729883)
- [Running deep feature flow for video recognition](https://github.com/ZHAOZHIHAO/RunningProgramms/tree/master/running_deep_feature_flow_for_video_recognition) DFF视频目标检测docker配置。
- [基于mxnet的Deep-Feature-Flow源码架构解析](https://www.jianshu.com/p/0f5e26611473) 也是一个实习生，研究了一段时间的视频目标检测，同时对DFF代码进行了一些尝试，可以作为参考。[菜鸟实习日记~day11(C3D+mxnet编译）](https://www.jianshu.com/p/0b4964261673)
- [视频检测分割--Deep Feature Flow for Video Recognition](https://blog.csdn.net/zhangjunhit/article/details/76665253)
- [Deep Feature Flow -CVPR 2017](https://blog.csdn.net/lxt1994/article/details/79952310)
- [论文笔记《Deep Feature Flow for Video Recognition》](http://www.xzhewei.com/Note-%E7%AC%94%E8%AE%B0/Video-Object-Detection/Note-Deep-Feature-Flow-for-Video-Recognition/) 较为详细的DFF论文解读笔记，主要是论文的中文翻译，可以参考。

