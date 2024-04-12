# FL-Simulator
Pytorch implementations of some general optimization methods in the federated learning community.

## Basic Methods

**FedAvg**: [Communication-Efficient Learning of Deep Networks
from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)

**FedProx**: [Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf)

**FedAdam**: [Adaptive Federated Optimization](https://openreview.net/pdf?id=LkFG3lB13U5)

**SCAFFOLD**: [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf)

**FedDyn**: [Federated Learning Based on
Dynamic Regularization](https://openreview.net/pdf?id=B7v4QMR6Z9w)

**FedCM**: [FedCM: Federated Learning with
Client-level Momentum](https://arxiv.org/pdf/2106.10874.pdf)

**FedSAM/MoFedSAM**: [Generalized Federated Learning via Sharpness Aware Minimization](https://proceedings.mlr.press/v162/qu22a/qu22a.pdf)

**FedGamma**: [Fedgamma: Federated learning with global sharpness-aware minimization](https://ieeexplore.ieee.org/abstract/document/10269141)

**FedSpeed**: [FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy](https://openreview.net/pdf?id=bZjxxYURKT)

**FedSMOO**: [Dynamic Regularized Sharpness Aware Minimization in Federated Learning: Approaching Global Consistency and Smooth Landscape](https://proceedings.mlr.press/v202/sun23h.html)


## Usage
### Training

FL-Simulator works on one single CPU/GPU to simulate the training process of federated learning (FL) with the PyTorch framework. If you want to train the centralized-FL with FedAvg method on the ResNet-18 and Cifar-10 dataset (10% active clients per round of total 100 clients, and heterogeneous dataset split is Dirichlet-0.6), you can use:
```train cFL
python train.py --non-iid --dataset CIFAR-10 --model ResNet18 --split-rule Dirichlet --split-coef 0.6 --active-ratio 0.1 --total-client 100
```
Other hyperparameters are introduced in the train.py file.

## How to implement your own method?

FL-Simulator pre-define the basic Server class and Client class, which are executed according to the vanilla $FedAvg$ algorithm. If you want define a new method, you can define a new server file first with:

- **process_for_communication( ):** how your method pre-processes the variables for communication to each client 

- **postprocess( ):** how your method post-processes the received variables from each local client

- **global_update( ):** how your method processes the update on the global model

Then you can define a new client file or a new local optimizer for your own method to perform the local training. Similarly, you can directly define a new server class to rebuild the inner-operations.



## Some Experiments
#### We show some results of the ResNet-18-GN model on the CIFAR-10 dataset. The corresponding hyperparameters are stated in the following. The time costs are tested on the NVIDIA® Tesla® V100 Tensor Core. 
<p align="center">
<table>
    <tbody align="center" valign="center">
        <tr>
            <td colspan="1">   </td>
            <td colspan="10"> CIFAR-10 (ResNet-18-GN) T=1000 </td>
        </tr>
        <tr>
            <td colspan="1">  </td>
            <td colspan="5">  10%-100 (bs=50 Local-epoch=5)  </td>
            <td colspan="5">  5%-200 (bs=25 Local-epoch=5)	 </td>
        </tr>
        <tr>
            <td colspan="1">  </td>
            <td colspan="1"> IID </td>
            <td colspan="1"> Dir-0.6 </td>
            <td colspan="1"> Dir-0.3 </td>
            <td colspan="1"> Dir-0.1 </td>
            <td colspan="1"> Time / round </td>
            <td colspan="1"> IID </td>
            <td colspan="1"> Dir-0.6 </td>
            <td colspan="1"> Dir-0.3 </td>
            <td colspan="1"> Dir-0.1 </td>
            <td colspan="1"> Time / round </td>
        </tr>
        <tr>
            <td colspan="1">   </td>
            <td colspan="10"> SGD basis </td>
        </tr>
        <tr>
            <td colspan="1"> FedAvg </td>
            <td colspan="1"> 82.52 </td>
            <td colspan="1"> 80.65 </td>
            <td colspan="1"> 79.75 </td>
            <td colspan="1"> 77.31 </td>
            <td colspan="1"> 15.86s </td>
            <td colspan="1"> 81.09 </td>
            <td colspan="1"> 79.93 </td>
            <td colspan="1"> 78.66 </td>
            <td colspan="1"> 75.21 </td>
            <td colspan="1"> 17.03s </td>
        </tr>
        <tr>
            <td colspan="1"> FedProx </td>
            <td colspan="1"> 82.54 </td>
            <td colspan="1"> 81.05 </td>
            <td colspan="1"> 79.52 </td>
            <td colspan="1"> 76.86 </td>
            <td colspan="1"> 19.78s </td>
            <td colspan="1"> 81.56 </td>
            <td colspan="1"> 79.49 </td>
            <td colspan="1"> 78.76 </td>
            <td colspan="1"> 75.84 </td>
            <td colspan="1"> 20.97s </td>
        </tr>
        <tr>
            <td colspan="1"> FedAdam </td>
            <td colspan="1"> 84.32 </td>
            <td colspan="1"> 82.56 </td>
            <td colspan="1"> 82.12 </td>
            <td colspan="1"> 77.58 </td>
            <td colspan="1"> 15.91s </td>
            <td colspan="1"> 83.29 </td>
            <td colspan="1"> 81.22 </td>
            <td colspan="1"> 80.22 </td>
            <td colspan="1"> 75.83 </td>
            <td colspan="1"> 17.67s </td>
        </tr>
        <tr>
            <td colspan="1"> SCAFFOLD </td>
            <td colspan="1"> 84.88 </td>
            <td colspan="1"> 83.53 </td>
            <td colspan="1"> 82.75 </td>
            <td colspan="1"> 79.92 </td>
            <td colspan="1"> 20.09s </td>
            <td colspan="1"> 84.24 </td>
            <td colspan="1"> 83.01 </td>
            <td colspan="1"> 82.04 </td>
            <td colspan="1"> 78.23 </td>
            <td colspan="1"> 22.21s </td>
        </tr>
        <tr>
            <td colspan="1"> FedDyn </td>
            <td colspan="1"> 85.46 </td>
            <td colspan="1"> 84.22 </td>
            <td colspan="1"> 83.22 </td>
            <td colspan="1"> 78.96 </td>
            <td colspan="1"> 20.82s </td>
            <td colspan="1"> 81.11 </td>
            <td colspan="1"> 80.25 </td>
            <td colspan="1"> 79.43 </td>
            <td colspan="1"> 75.43 </td>
            <td colspan="1"> 22.68s </td>
        </tr>
        <tr>
            <td colspan="1"> FedCM </td>
            <td colspan="1"> 85.74 </td>
            <td colspan="1"> 83.81 </td>
            <td colspan="1"> 83.44 </td>
            <td colspan="1"> 78.92 </td>
            <td colspan="1"> 20.74s </td>
            <td colspan="1"> 83.77 </td>
            <td colspan="1"> 82.01 </td>
            <td colspan="1"> 80.77 </td>
            <td colspan="1"> 75.91 </td>
            <td colspan="1"> 21.24s </td>
        </tr>
        <tr>
            <td colspan="1">   </td>
            <td colspan="10"> SAM basis </td>
        </tr>
        <tr>
            <td colspan="1"> FedGamma </td>
            <td colspan="1"> 85.74 </td>
            <td colspan="1"> 84.80 </td>
            <td colspan="1"> 83.81 </td>
            <td colspan="1"> 80.72 </td>
            <td colspan="1"> 30.13s </td>
            <td colspan="1"> 84.99 </td>
            <td colspan="1"> 84.02 </td>
            <td colspan="1"> 83.03 </td>
            <td colspan="1"> 80.09 </td>
            <td colspan="1"> 33.63s </td>
        </tr>
        <tr>
            <td colspan="1"> MoFedSAM </td>
            <td colspan="1"> 87.24 </td>
            <td colspan="1"> 85.74 </td>
            <td colspan="1"> 85.14 </td>
            <td colspan="1"> 81.58 </td>
            <td colspan="1"> 29.06s </td>
            <td colspan="1"> 86.27 </td>
            <td colspan="1"> 84.71 </td>
            <td colspan="1"> 83.44 </td>
            <td colspan="1"> 79.02 </td>
            <td colspan="1"> 32.45s </td>
        </tr>
        <tr>
            <td colspan="1"> FedSpeed </td>
            <td colspan="1"> 87.31 </td>
            <td colspan="1"> 86.33 </td>
            <td colspan="1"> 85.39 </td>
            <td colspan="1"> 82.26 </td>
            <td colspan="1"> 29.48s </td>
            <td colspan="1"> 86.87 </td>
            <td colspan="1"> 85.07 </td>
            <td colspan="1"> 83.94 </td>
            <td colspan="1"> 79.66 </td>
            <td colspan="1"> 33.69s </td>
        </tr>
        <tr>
            <td colspan="1"> FedSMOO </td>
            <td colspan="1"> 87.70 </td>
            <td colspan="1"> 86.87 </td>
            <td colspan="1"> 86.04 </td>
            <td colspan="1"> 83.30 </td>
            <td colspan="1"> 30.43s </td>
            <td colspan="1"> 87.40 </td>
            <td colspan="1"> 85.97 </td>
            <td colspan="1"> 85.14 </td>
            <td colspan="1"> 81.35 </td>
            <td colspan="1"> 34.80s </td>
        </tr>
    </tbody>
</table>
</p>

The blank parts are awaiting updates.

**Some key hyparameters selection**
<p align="center">
<table>
    <tbody align="center" valign="center">
        <tr>
            <td colspan="1">  </td>
            <td colspan="1"> local Lr </td>
            <td colspan="1"> global Lr </td>
            <td colspan="1"> Lr decay </td>
            <td colspan="1"> SAM Lr </td>
            <td colspan="1"> proxy coefficient </td>
            <td colspan="1"> client-momentum coefficiet </td>
        </tr>
        <tr>
            <td colspan="1"> FedAvg </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> 1.0 </td>
            <td colspan="1"> 0.998 </td>
            <td colspan="1"> - </td>
            <td colspan="1"> - </td>
            <td colspan="1"> - </td>
        </tr>
        <tr>
            <td colspan="1"> FedProx </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> 1.0 </td>
            <td colspan="1"> 0.998 </td>
            <td colspan="1"> - </td>
            <td colspan="1"> 0.1 / 0.01 </td>
            <td colspan="1"> - </td>
        </tr>
        <tr>
            <td colspan="1"> FedAdam </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> 0.1 / 0.05 </td>
            <td colspan="1"> 0.998 </td>
            <td colspan="1"> - </td>
            <td colspan="1"> - </td>
            <td colspan="1"> - </td>
        </tr>
        <tr>
            <td colspan="1"> SCAFFOLD </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> 1.0 </td>
            <td colspan="1"> 0.998 </td>
            <td colspan="1"> - </td>
            <td colspan="1"> - </td>
            <td colspan="1"> - </td>
        </tr>
        <tr>
            <td colspan="1"> FedDyn </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> 1.0 </td>
            <td colspan="1"> 0.9995 / 1.0 </td>
            <td colspan="1"> - </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> - </td>
        </tr>
        <tr>
            <td colspan="1"> FedCM </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> 1.0 </td>
            <td colspan="1"> 0.998 </td>
            <td colspan="1"> - </td>
            <td colspan="1"> - </td>
            <td colspan="1"> 0.1 </td>
        </tr>
        <tr>
            <td colspan="1"> FedGamma </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> 1.0 </td>
            <td colspan="1"> 0.998 </td>
            <td colspan="1"> 0.01 </td>
            <td colspan="1"> - </td>
            <td colspan="1"> - </td>
        </tr>
        <tr>
            <td colspan="1"> MoFedSAM </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> 1.0 </td>
            <td colspan="1"> 0.998 </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> - </td>
            <td colspan="1"> 0.05 / 0.1 </td>
        </tr>
        <tr>
            <td colspan="1"> FedSpeed </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> 1.0 </td>
            <td colspan="1"> 0.998 </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> - </td>
        </tr>
        <tr>
            <td colspan="1"> FedSMOO </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> 1.0 </td>
            <td colspan="1"> 0.998 </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> 0.1 </td>
            <td colspan="1"> - </td>
        </tr>
    </tbody>
</table>
</p>

The hyperparameter selections above are for reference only. Each algorithm has unique properties to match the corresponding hyperparameters. In order to facilitate a relatively fair comparison, we report a set of selections that each method can perform well in general cases. Please adjust the hyperparameters according to changes in the different model backbones and datasets.

## ToDo
- [ ] Decentralized Implementation
- [ ] Delayed / Asynchronous Implementation
- [x] Hyperparameter Selections
- [x] Related Advances (Long-Term)


## Citation
If this codebase can help you, please cite our papers: 

[FedSpeed](https://arxiv.org/abs/2302.10429) (ICLR 2023):
```bibtex
@article{sun2023fedspeed,
  title={Fedspeed: Larger local interval, less communication round, and higher generalization accuracy},
  author={Sun, Yan and Shen, Li and Huang, Tiansheng and Ding, Liang and Tao, Dacheng},
  journal={arXiv preprint arXiv:2302.10429},
  year={2023}
}
```
[FedSMOO](https://proceedings.mlr.press/v202/sun23h.html) (ICML 2023 Oral):
```bibtex
@inproceedings{sun2023dynamic,
  title={Dynamic regularized sharpness aware minimization in federated learning: Approaching global consistency and smooth landscape},
  author={Sun, Yan and Shen, Li and Chen, Shixiang and Ding, Liang and Tao, Dacheng},
  booktitle={International Conference on Machine Learning},
  pages={32991--33013},
  year={2023},
  organization={PMLR}
}
```