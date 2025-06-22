import pandas as pd
import numpy as np
from typing import Union

class MySVM:
    def __init__(self, n_iter: int = 10, learning_rate: float = 0.1, metric: Union[None, str] = None, reg: Union[None, str] = None, l1_coef: float = 0.0, l2_coef: float = 0.0, sgd_sample: Union[int, float, None] = None, random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.b = None
        self.metric = metric
        self.best_metric = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X_input: pd.DataFrame, Y_input: pd.Series, verbose: Union[bool, int] = False):
        Y = self._preprocess_labels(Y_input)
        X = self._preprocess_features(X_input)
        self.weights = np.zeros((X_input[0], 1)) + 1
        self.b = 1
        iter = 0
        
        while iter < self.n_iter:
            loss = self._calc_loss(X, Y)
            iter += 1
        
    def _preprocess_labels(self, Y_input: pd.Series) -> np.array:
        pos, neg = Y_input.unique()
        Y_preprocessed = pd.Series(np.zeros(Y_input.shape))
        Y_preprocessed[Y_input == pos] = 1
        Y_preprocessed[Y_input == neg] = -1
        return Y_preprocessed.to_numpy()
    
    def _preprocess_features(self, X_input: pd.DataFrame) -> np.array:
        return X_input.to_numpy()
    
    def _calc_loss(self, X: np.array, Y: np.array) -> float:
        
        