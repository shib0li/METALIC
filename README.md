# *METALIC*: Meta Learning of Interface Conditions for Multi-Domain Physics-Informed Neural Networks

by [Shibo Li*](https://imshibo.com), Michael Penwarden*, Yiming Xu, Conor Tillinghast, Akil Narayan, [Mike Kirby](https://www.cs.utah.edu/~kirby/) and [Shandian Zhe](https://www.cs.utah.edu/~zhe/)

<p align="center">
    <br>
    <img src="images/1582-Thumb.png" width="600" />
    <br>
<p>

<h4 align="center">
    <p>
        <a href="https://openreview.net/forum?id=e694Xvz6Q6">Paper</a> |
        <a href="https://github.com/shib0li/METALIC/blob/main/images/slides.pdf">Slides</a> |
        <a href="https://github.com/shib0li/METALIC/blob/main/images/1582-poster-resize.png">Poster</a> 
    <p>
</h4>


Physics-informed neural networks (PINNs) are emerging as popular mesh-free solvers for partial differential equations (PDEs). Recent extensions decompose the domain, apply different PINNs to solve the problem in each subdomain, and stitch the subdomains at the interface. Thereby, they can further alleviate the problem complexity, reduce the computational cost, and allow parallelization. However, the performance of multi-domain PINNs is sensitive to the choice of the interface conditions. While quite a few conditions have been proposed, there is no suggestion about how to select the conditions according to specific problems. To address this gap, we propose META Learning of Interface Conditions (METALIC), a simple, efficient yet powerful approach to dynamically determine appropriate interface conditions for solving a family of parametric PDEs. Specifically, we develop two contextual multi-arm bandit (MAB) models. The first one applies to the entire training course, and online updates a Gaussian process (GP) reward that given the PDE parameters and interface conditions predicts the performance. We prove a sub-linear regret bound for both UCB and Thompson sampling, which in theory guarantees the effectiveness of our MAB. The second one partitions the training into two stages, one is the stochastic phase and the other deterministic phase; we update a GP reward for each phase to enable different condition selections at the two stages to further bolster the flexibility and performance. We have shown the advantage of METALIC on four bench-mark PDE families.

<!-- IFC-ODE $^2$ /GPT -->

# System Requirements

We highly recommend to use Docker to run our code. We have attached the docker build file `env.Dockerfile`. Or feel free to install the packages with pip/conda that could be found in the docker file.


# Run

To run 
```
python run_mab.py -domain=$DOMAIN -heuristic=$HEURISTIC -mode=$MODE -num_adam=$NUM_ADAM -int_adam=$ADAM_EPOCHS -num_lbfgs=$NUM_LBFGS -int_lbfgs=LBFGS_ITERS -int_adam_test=$TEST_ADAM_EPOCHS -int_lbfgs_test=$TEST_LGBFS_ITERS -device=$DEVICE

```


* `$DOMAIN` KdV, Burgers, Advec, 
* `$HEURISTIC` ucb, ts
* `$MODE` query, step
* `$NUM_ADAM` number of Adam stages
* `$ADAM_EPOCHS` number of epochs per Adam stage
* `$NUM_LBFGS` number of L-BFGS stages
* `$LBFGS_ITERS` max iters per L-BFGS stage
* `$TEST_ADAM_EPOCHS` test adam epochs
* `$TEST_LGBFS_ITERS` test L-BFGS max iters
* `$DEVICE` device to run

    


# License

METALIC is released under the MIT License, please refer the LICENSE for details

# Getting Involved
Feel free to submit Github issues or pull requests. Welcome to contribute to our project!

To contact us, never hestitate to send an email to `shibo@cs.utah.edu` or `shiboli.cs@gmail.com` 
<br></br>


# Citation
Please cite our paper if you find it helpful :)

```

@InProceedings{pmlr-v202-li23w,
  title = 	 {Meta Learning of Interface Conditions for Multi-Domain Physics-Informed Neural Networks},
  author =       {Li, Shibo and Penwarden, Michael and Xu, Yiming and Tillinghast, Conor and Narayan, Akil and Kirby, Mike and Zhe, Shandian},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {19855--19881},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/li23w/li23w.pdf},
  url = 	 {https://proceedings.mlr.press/v202/li23w.html},}

```
<br></br>