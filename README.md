
# Effective and Robust Adversarial Training Against Data and Label Corruptions

This is the official PyTorch repository for the implementation of [ERAT](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10517640).

## Requirements
Python3

PyTorch (> 1.0)

## Prepare datasets
1. Download datasets
```console
python downloaddata.py
```
2. Generate perturbed training data
The code for generating perturbed data can be accessible in [Delusive-Adversary](https://github.com/TLMichael/Delusive-Adversary), [DeepConfuse](https://github.com/kingfengji/DeepConfuse), [Unlearnable-Examples](https://github.com/HanxunH/Unlearnable-Examples).

## Train
```console
python Dual_main.py
```

If you find this code helpful for your research, please consider citing our [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10517640):
```
@article{zhang2024effective,
  title={Effective and Robust Adversarial Training Against Data and Label Corruptions},
  author={Zhang, Peng-Fei and Huang, Zi and Xu, Xin-Shun and Bai, Guangdong},
  journal={IEEE Transactions on Multimedia},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledge
Some of our code and datasets are based on [DivideMix](https://github.com/LiJunnan1992/DivideMix).
