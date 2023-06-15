# Iteration Learned
Iteration learned is a metric of example difficulty.
It is proposed in the paper: [An Empirical Study of Example Forgetting during Deep Neural Network Learning](https://arxiv.org/abs/1812.05159).

Bibtex: 
```
@inproceedings{Forgetting,
    title={An Empirical Study of Example Forgetting during Deep Neural Network Learning},
    author={Toneva, Mariya and Sordoni, Alessandro and Combes, Remi Tachet des and Trischler, Adam and Bengio, Yoshua and Gordon, Geoffrey J},
    booktitle={ICLR},
    year={2019}
}
```
It is defined as: 

A data point is said to be learned by a classifier at training iteration $t = \tau$ if the predicted
class at iteration $t = \tau − 1$ is different from the final prediction of the converged network and the
predictions at all iterations $t ≥ tau$ are equal to the final prediction of the converged network. Data
points consistently classified after all training steps and at the moment of initialization, are said to be
learned in step t = 0.