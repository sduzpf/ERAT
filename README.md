#This is the official pytorch repository for the implementation of ERAT
Getting Started

#Requirements
Python3
PyTorch (> 1.0)

1. Download datasets
python downloaddata.py

2. Generate perturbed training data
The code for generating perturbed data can be accessible in [Delusive-Adversary](https://github.com/TLMichael/Delusive-Adversary), [DeepConfuse](https://github.com/kingfengji/DeepConfuse), [Unlearnable-"Examples](https://github.com/HanxunH/Unlearnable-Examples).

#Train
python Dual_main.py

If you find this code helpful for your research, please consider to cite our [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10517640):
@ARTICLE{10517640,
  title={Effective and Robust Adversarial Training Against Data and Label Corruptions}, 
  author={Zhang, Peng-Fei and Huang, Zi and Xu, Xin-Shun and Bai, Guangdong},
  journal={IEEE Transactions on Multimedia}, 
  year={2024},
  pages={1-12},
}

Acknowledge
Some of our code and datasets are based on [DivideMix](https://github.com/LiJunnan1992/DivideMix).
