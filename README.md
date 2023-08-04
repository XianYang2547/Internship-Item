<p align="left">
  <a href [https://github.com/XianYang2547/Home-Page]">
  <img src="https://img.shields.io/badge/Author-@XianYang-000000.svg?logo=GitHub" alt="GitHub"></a>

  总体描述就是：下载数据集，进行处理（直方图均衡化，旋转等），使用yolov5检测出手部骨头，然后进一步筛选。使用143M的那个数据集训练9个小模型（残差18），将筛选出的骨节送入小模型计算等级得分，得出骨龄。做了简单的PYQT界面，选择本地图像，选择性别，后台加载模型，显示出检测结果。
  
### 下载数据和模型(https://pan.baidu.com/s/1VLb-wDaLkiQQA-DD_zyQEA?pwd=2547)
### 流程如下
  ![image](img/6.png)
### 结果
  ![image](img/20230802115911.jpg)
### PYQT界面
  ![image](img/图片1.png)
                    




