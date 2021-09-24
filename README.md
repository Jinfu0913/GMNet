# GMNet-
Code and result about GMNet(IEEE TIP)<br>
'GMNet: Graded-Feature Multilabel-Learning Network for RGB-Thermal Urban Scene Semantic Segmentation' [IEEE TIP](https://ieeexplore.ieee.org/document/9531449)
![image](https://user-images.githubusercontent.com/65142166/134632191-8cfbc175-27e8-43de-b72b-bd5bf8f7c9b9.png)

# Requirements
Python 3.6.5, Pytorch 1.8.1+, Cuda 10.2, TensorboardX 2.0, opencv-python

# Dataset and Evaluate tools
RGB-D SOD Datasets can be found in:  http://dpfan.net/d3netbenchmark/  or https://github.com/jiwei0921/RGBD-SOD-datasets <br>

we use the matlab verison provide by Dengping Fan, we provide our test datesets [百度网盘](https://pan.baidu.com/s/1tVJCWRwqIoZQ3KAplMSHsA) 提取码：zust 

# Result
Test maps: [百度网盘](https://pan.baidu.com/s/1QcEAHlS8llyX-i3kX4npAA)  提取码：zust <br>
Pretrained model download:[百度网盘](https://pan.baidu.com/s/1reGFvIYX7rZjzKuaDcs-3A)  提取码：zust <br>
PS: we resize the testing data to the size of 224 * 224 for quicky evaluate, [百度网盘](https://pan.baidu.com/s/1t5cES-RAnMCLJ76s9bwzmA)  提取码：zust <br>

# Citation
@ARTICLE{9424966,<br>
  author={Zhou, Wujie and Zhu, Yun and Lei, Jingsheng and Wan, Jian and Yu, Lu},<br>
  journal={IEEE Transactions on Multimedia}, <br>
  title={CCAFNet: Crossflow and Cross-scale Adaptive Fusion Network for Detecting Salient Objects in RGB-D Images}, <br>
  year={2021},<br>
  doi={10.1109/TMM.2021.3077767}}<br>

# Acknowledgement
The implement this project based on the code of ‘Cascaded Partial Decoder for Fast and Accurate Salient Object Detection, CVPR2019’and 'BBS-Net: RGB-D Salient Object Detection with a Bifurcated Backbone Strategy Network' proposed by Wu et al and Deng et al.

# Contact
Please drop me an email for further problems or discussion: zzzyylink@gmail.com or wujiezhou@163.com
