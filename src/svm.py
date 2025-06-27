import pandas as pd
import numpy as np
import random
from typing import Union, Tuple

class MySVM:
    def __init__(self, n_iter: int = 10, learning_rate: float = 0.001, c: float = 1.0, sgd_sample: Union[float, int, None] = None, random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.b = None
        self.c = c
        self.random_state = random_state
        self.sgd_sample = sgd_sample

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X_input: pd.DataFrame, Y_input: pd.Series, verbose: Union[bool, int] = False):
        Y = self._preprocess_labels(Y_input)
        X = self._preprocess_features(X_input)
        self.weights = np.ones((X_input.shape[1], 1))
        self.b = 1.0
        iter = 0

        samples_count = Y.shape[0]
        if self.sgd_sample is not None:
            random.seed(self.random_state)
            samples_count = round(self.sgd_sample * Y.shape[0]) if isinstance(self.sgd_sample, float) else self.sgd_sample
        
        while iter < self.n_iter:
            X_batch = X
            Y_batch = Y
            if self.sgd_sample is not None:
                samples = random.sample(range(Y.shape[0]), samples_count)
                X_batch = X[samples]
                Y_batch = Y[samples]
                
            for i in range(samples_count):
                X_i = np.array([X_batch[i, :]])
                Y_i = float(Y_batch[i])
                Y_predict = float(self._predict(X_i, Y_i))
                
                gradient_weights, gradient_bias = self._calc_gradient(X_i, Y_i, Y_predict)
                self.weights -= self.learning_rate * gradient_weights
                self.b -= self.learning_rate * gradient_bias
            
            loss = self._calc_loss(X, Y)
            if (verbose > 0) and (iter % verbose == 0):
                ind_log = 'start' if iter == 0 else iter
                print(f'ind_log | loss: {loss}')
            iter += 1
        
    def _preprocess_labels(self, Y_input: pd.Series) -> np.array:
        Y_preprocessed = pd.Series(np.zeros(Y_input.shape))
        Y_preprocessed[Y_input == 1] = 1
        Y_preprocessed[Y_input == 0] = -1
        return Y_preprocessed.to_numpy().reshape(-1, 1)
    
    def _preprocess_features(self, X_input: pd.DataFrame) -> np.array:
        return X_input.to_numpy()
    
    def _predict(self, X: np.array, Y: np.array) -> np.array:
        return Y * (X @ self.weights + self.b)
    
    def _calc_loss(self, X: np.array, Y: np.array) -> float:
        predict = self._predict(X, Y)
        wrong_ind = predict < 1
        loss = np.sum(self.weights**2)

        if sum(wrong_ind):
            loss += self.c * np.sum(1 - predict[wrong_ind]) / Y.shape[0]
        return loss
    
    def _calc_gradient(self, X: np.array, Y: np.array, Y_predict: np.array) -> Tuple[np.array, float]:
        grad_weights = 2 * self.weights
        grad_bias = 0
        
        if  Y_predict < 1:
            grad_weights -= self.c * X.T * Y
            grad_bias -= self.c * Y
        return grad_weights, grad_bias
    
    def get_coef(self):
        return self.weights.ravel(), self.b
    
    def predict(self, X_input: pd.DataFrame):
        X = self._preprocess_features(X_input)
        predict = np.array(np.sign((X @ self.weights + self.b)), dtype=int)
        predict[predict == -1] = 0
        return predict.ravel()