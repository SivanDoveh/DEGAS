# SEGAN: Searching  Efficiently for new GANs
This project is for reproducing the results of paper SEGAN: Searching  Efficiently for new GANs

## Prerequisites
- Python 3.5.5,TensorFlow 1.4.0, NumPy, SciPy, Sklearn

- To reproduce the IS and FID run:
  python train_gan.py dataset 'results/' labels --arch='arch_name'  --gpu 1 --seed 1
    - labels= 'unsup' or 'sup'

    - For example, cifar10 evaluation:
    
      python train_gan.py cifar10 'results/' 'unsup' --arch='cifar10_n1_resnet_const_end_3e1_no_tg_200'  --gpu 1 --seed 1
