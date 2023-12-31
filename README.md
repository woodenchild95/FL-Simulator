# FL-Simulator
Pytorch implementations of some general federated optimization methods.

## Basic Methods

$FedAvg $: [Communication-Efficient Learning of Deep Networks
from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)

$FedProx$: [Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf)

$FedAdam$: [Adaptive Federated Optimization](https://openreview.net/pdf?id=LkFG3lB13U5)

$SCAFFOLD$: [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf)

$FedDyn$: [Federated Learning Based on
Dynamic Regularization](https://openreview.net/pdf?id=B7v4QMR6Z9w)

$FedCM$: [FedCM: Federated Learning with
Client-level Momentum](https://arxiv.org/pdf/2106.10874.pdf)

$FedSAM/MoFedSAM$: [Generalized Federated Learning via Sharpness Aware Minimization](https://proceedings.mlr.press/v162/qu22a/qu22a.pdf)

$FedSpeed$: [FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy](https://openreview.net/pdf?id=bZjxxYURKT)


## Usage
### Training

FL-Simulator works on one single CPU/GPU to simulate the training process of federated learning (FL) with the PyTorch framework. If you want to train the centralized-FL with FedAvg method on the ResNet-18 and Cifar-10 dataset (10% active clients per round of total 100 clients, and heterogeneous dataset split is Dirichlet-0.6), you can use:
```train cFL
python train.py --non-iid --dataset CIFAR-10 --model ResNet18 --split-rule Dirichlet --split-coef 0.6 --active-ratio 0.1 --total-client 100
```
Other hyperparameters are introduced in the train.py file.

## How to create new?

FL-Simulator pre-define the basic Server class and Client class, which are executed according to the vanilla $FedAvg$ algorithm. If you want define a new method, you can define a new server file first with:

- process_for_communication( ):
    
        how your method preprocesses the communication variables

- global_update( ):

        how your method processes the update on the global model

- postprocess( ):

        how your method processes the received variables from local clients

Then, you can define a new client file for this new method.

## Experiments
#### We train the ResNet-18-GN on the CIFAR-10 dataset and test the global model. 
<p align="center">
<table>
    <tbody align="center" valign="center">
        <tr>
            <td colspan="1">   </td>
            <td colspan="8"> CIFAR-10 (ResNet-18-GN) T=1000 </td>
        </tr>
        <tr>
            <td colspan="1">  </td>
            <td colspan="4">  10%-100 (bs=50 Local-epoch=5)  </td>
            <td colspan="4">  5%-200 (bs=25 Local-epoch=5)	 </td>
        </tr>
        <tr>
            <td colspan="1">  </td>
            <td colspan="1"> IID </td>
            <td colspan="1"> Dir-0.6 </td>
            <td colspan="1"> Dir-0.3 </td>
            <td colspan="1"> Dir-0.1 </td>
            <td colspan="1"> IID </td>
            <td colspan="1"> Dir-0.6 </td>
            <td colspan="1"> Dir-0.3 </td>
            <td colspan="1"> Dir-0.1 </td>
        </tr>
        <tr>
            <td colspan="1"> FedAvg </td>
            <td colspan="1"> 82.52 </td>
            <td colspan="1"> 80.65 </td>
            <td colspan="1"> 79.75 </td>
            <td colspan="1"> 77.31 </td>
            <td colspan="1"> 81.09 </td>
            <td colspan="1"> 79.93 </td>
            <td colspan="1"> 78.66 </td>
            <td colspan="1"> 75.21 </td>
        </tr>
        <tr>
            <td colspan="1"> FedProx </td>
            <td colspan="1"> 82.54 </td>
            <td colspan="1"> 81.05 </td>
            <td colspan="1"> 79.52 </td>
            <td colspan="1"> 76.86 </td>
            <td colspan="1"> 81.56 </td>
            <td colspan="1"> 79.49 </td>
            <td colspan="1"> 78.76 </td>
            <td colspan="1"> 75.84 </td>
        </tr>
        <tr>
            <td colspan="1"> FedAdam </td>
            <td colspan="1"> 84.32 </td>
            <td colspan="1"> 82.56 </td>
            <td colspan="1"> 82.12 </td>
            <td colspan="1"> 77.58 </td>
            <td colspan="1"> 83.29 </td>
            <td colspan="1"> 81.22 </td>
            <td colspan="1"> 80.22 </td>
            <td colspan="1"> 75.83 </td>
        </tr>
        <tr>
            <td colspan="1"> SCAFFOLD </td>
            <td colspan="1"> 84.88 </td>
            <td colspan="1"> 83.53 </td>
            <td colspan="1"> 82.75 </td>
            <td colspan="1"> 79.92 </td>
            <td colspan="1"> 84.24 </td>
            <td colspan="1"> 83.01 </td>
            <td colspan="1"> 82.04 </td>
            <td colspan="1"> 78.23 </td>
        </tr>
        <tr>
            <td colspan="1"> FedDyn </td>
            <td colspan="1"> 85.46 </td>
            <td colspan="1"> 84.22 </td>
            <td colspan="1"> 83.22 </td>
            <td colspan="1"> 78.96 </td>
            <td colspan="1"> 81.11 </td>
            <td colspan="1"> 80.25 </td>
            <td colspan="1"> 79.43 </td>
            <td colspan="1"> 75.43 </td>
        </tr>
        <tr>
            <td colspan="1"> FedCM </td>
            <td colspan="1"> 85.74 </td>
            <td colspan="1"> 83.81 </td>
            <td colspan="1"> 83.44 </td>
            <td colspan="1"> 78.92 </td>
            <td colspan="1"> 83.77 </td>
            <td colspan="1"> 82.01 </td>
            <td colspan="1"> 80.77 </td>
            <td colspan="1"> 75.91 </td>
        </tr>
        <tr>
            <td colspan="1"> MoFedSAM </td>
            <td colspan="1"> 87.24 </td>
            <td colspan="1"> 85.74 </td>
            <td colspan="1"> 85.14 </td>
            <td colspan="1"> 81.58 </td>
            <td colspan="1"> 86.27 </td>
            <td colspan="1"> 84.71 </td>
            <td colspan="1"> 83.44 </td>
            <td colspan="1"> 79.02 </td>
        </tr>
        <tr>
            <td colspan="1"> FedSpeed </td>
            <td colspan="1"> 87.72 </td>
            <td colspan="1"> 86.05 </td>
            <td colspan="1"> 85.25 </td>
            <td colspan="1"> 82.05 </td>
            <td colspan="1"> 86.87 </td>
            <td colspan="1"> 85.07 </td>
            <td colspan="1"> 83.94 </td>
            <td colspan="1"> 79.66 </td>
        </tr>
    </tbody>
</table>
</p>


## ToDo
- [ ] Decentralized FL Implementation.


## Citation
If this codebase can help you, please cite our paper [FedSpeed](https://arxiv.org/abs/2302.10429):
```bibtex
@article{sun2023fedspeed,
  title={Fedspeed: Larger local interval, less communication round, and higher generalization accuracy},
  author={Sun, Yan and Shen, Li and Huang, Tiansheng and Ding, Liang and Tao, Dacheng},
  journal={arXiv preprint arXiv:2302.10429},
  year={2023}
}
```
