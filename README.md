# 欢迎你与我联系！正在飞速完善仓库中🏃......

## **配置环境**
```bash
conda env create -f environment.yml
```

## **LabNet的测试数据集和预训练模型**  
通过网盘分享的文件：test_datasets
链接: https://pan.baidu.com/s/13JjegeIgyoLb5Db9zmKXIw?pwd=hhxx 提取码: hhxx

通过网盘分享的文件：pretrain_model
链接: https://pan.baidu.com/s/1TRrABPW4hO2JoQ5zGWlw_g?pwd=hhxx 提取码: hhxx


## **LabNet的测试指令**
#### LabNetx4模型
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -u /share/home/104825/jiancong/LabNet/codes/config/LabNet/test.py -opt=/share/home/104825/jiancong/LabNet/codes/config/LabNet/options/setting1/test/test_setting1_x4.yml
```

#### LabNetx3模型
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -u /share/home/104825/jiancong/LabNet/codes/config/LabNet/test.py -opt=/share/home/104825/jiancong/LabNet/codes/config/LabNet/options/setting1/test/test_setting1_x3.yml
```

#### LabNetx2模型
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -u /share/home/104825/jiancong/LabNet/codes/config/LabNet/test.py -opt=/share/home/104825/jiancong/LabNet/codes/config/LabNet/options/setting1/test/test_setting1_x2.yml
```

## **LabNet的训练数据集下载地址** 
#### **DIV2K**
https://data.vision.ee.ethz.ch/cvl/DIV2K/
#### **Flickr2K**
http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar

## **LabNet的训练指令**
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -u /share/home/104632/jiancong/LabNet/codes/config/LabNet/train.py -opt=/share/home/104632/jiancong/LabNet/codes/config/LabNet/options/setting1/train/train_setting1_x4.yml
```

## **RealNet的测试指令**
```bash
xxx
```

## **联系方式**  
### **wechat**
monakasen

### **QQ**
851805291


## **致谢**
特别感谢代码仓库：https://github.com/greatlog/DAN  
我们的仓库是基于以上仓库进行修改，如果使用到本仓库，也请对以下论文进行引用：  
```bash
@article{luo2020unfolding,
  title={Unfolding the Alternating Optimization for Blind Super Resolution},
  author={Luo, Zhengxiong and Huang, Yan and Li, Shang and Wang, Liang and Tan, Tieniu},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={33},
  year={2020}
}
@misc{luo2021endtoend,
      title={End-to-end Alternating Optimization for Blind Super Resolution}, 
      author={Zhengxiong Luo and Yan Huang and Shang Li and Liang Wang and Tieniu Tan},
      year={2021},
      eprint={2105.06878},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

