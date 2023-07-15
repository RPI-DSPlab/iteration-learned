# Iteration (Epoch) Learned, Prediction Depth, and Consistency Score

THis repo consists of the implementation of the metrics: **Iteration (Epoch) Learned**, **Prediction Depth**, and **Consistency Score**.
All of these are built on the same structure, which is to predict the same dataset with the same model and store the 
score in the same format for better comparison.

`dadtaset.py` has dataset loaders for CIFAR10

`model.py` has the implementation of vgg16

# Iteration (Epoch) Learned

The corresponding files are `il_main.py` `il_util.py` `il_config.py`.

Iteration (Epoch) learned is a metric of example difficulty.
It is proposed in the paper: [An Empirical Study of Example Forgetting during Deep Neural Network Learning](https://arxiv.org/abs/1812.05159) by the authors defining **Forgetting and learning events**.

**Iteration Learned** is defined as a separated metric in hte paper: [Deep Learning Through the Lens of Example Difficulty](https://arxiv.org/abs/2106.09647).

Bibtex: 
```
@inproceedings{Forgetting,
    title={An Empirical Study of Example Forgetting during Deep Neural Network Learning},
    author={Toneva, Mariya and Sordoni, Alessandro and Combes, Remi Tachet des and Trischler, Adam and Bengio, Yoshua and Gordon, Geoffrey J},
    booktitle={ICLR},
    year={2019}
}
```
# **Iteration Learned** is defined as: 

> A data point is said to be learned by a classifier at training iteration $t = \tau$ if the predicted
class at iteration $t = \tau − 1$ is different from the final prediction of the converged network and the
predictions at all iterations $t ≥ \tau$ are equal to the final prediction of the converged network. Data
points consistently classified after all training steps and at the moment of initialization, are said to be
learned in step $t = 0$.


# Implementation:

This is implemented by: 

1. We train choose the model and the dataset to test with
2. For every iteration (or epoch) during training (`train()`), we use the cureent (not fully trained) model to predict images from the dataset and store the correctness of this prediction
3. Follow the defination of **Iteration Learned**, we determine the iteration (or epoch) learned.
