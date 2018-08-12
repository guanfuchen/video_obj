# Seq-NMS for Video Object Detection

| 会议／期刊 | 作者 | 论文 |
| ---- | ---- | ---- |
| arXiv 2016 | KHan W, Khorrami P, Paine T L | Seq-NMS for Video Object Detection |

---
## 核心描述

```
One problem we noticed with Faster R-CNN on the validation set was that non-maximum suppression
(NMS) frequently chose the wrong bounding box after object classification. It would choose boxes
that were overly large, resulting in a smaller intersection-over-union (IoU) with the ground truth box
because the union of areas was large. The large boxes often had very high object scores, possibly
because more information is available to be extracted during RoI pooling.

We assume that neighboring frames should have similar objects, and their bounding boxes should be 
similar in position and size, i.e. temporal consistency.

Seq-NMS has three steps: Step 1) Sequence Selection, Step 2) Sequence Re-scoring, Step 3) Suppression.
```

---
## 主要步骤

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/seq_nms_arch.png)

### Sequence Selection

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/seq_nms_seq_selection_arch.png)

### Sequence Re-scoring

### Suppression


---
## 代码阅读

代码主要参考daijifeng FGFA相关的代码实现[seq_nms.py](https://github.com/guanfuchen/Flow-Guided-Feature-Aggregation/tree/master/lib/nms/seq_nms.py)，以及另一个相似的实现代码[seqnms.py](https://github.com/lrghust/Seq-NMS/blob/master/seqnms.py)。

---
## 参考资料
- [论文笔记《Seq-nms for video object detection》](http://www.xzhewei.com/Note-%E7%AC%94%E8%AE%B0/Video-Object-Detection/Note-Seq-nms-for-video-object-detection/)