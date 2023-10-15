# VRI

The official PyTorch implemention of the paper "Variational Rectification Inference for Learning with Noisy Labels".


Resnet-18, run 
<code>python main_augmc_mixup_preres.py --corruption_type <i>unif</i> --corruption_prob <i>0.4</i> --dataset <i>cifar10</i> --gpuid <i>0</i></code>

WideResnet-28-10, run 
<code>python main_augmc_mixup_wres.py --corruption_type <i>unif</i> --corruption_prob <i>0.4</i> --dataset <i>cifar10</i> --gpuid <i>0</i></code>
