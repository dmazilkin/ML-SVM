import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Tuple

from src.svm import MySVM

N = 500

def create_data() -> Tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    
    X = pd.DataFrame(np.zeros((N, 2)) + 10 * np.random.rand(N, 2))
    Y = (X.loc[:, 0] > X.loc[:, 1] - 2).astype(int)
    
    return X, Y

def main():
    X, Y = create_data()
    
    model = MySVM(n_iter=10)
    model.fit(X, Y, verbose=True)
    Y_predicted = model.predict(X)
    
    figure, axis = plt.subplots(1, 2)
    axis[0].scatter(X.to_numpy()[Y == 1][:, 0], X.to_numpy()[Y == 1][:, 1], color='blue')
    axis[0].scatter(X.to_numpy()[Y == 0][:, 0], X.to_numpy()[Y == 0][:, 1], color='red')
    axis[0].set_title('Real data')
    axis[1].scatter(X.to_numpy()[Y_predicted == 1][:, 0], X.to_numpy()[Y_predicted == 1][:, 1], color='blue')
    axis[1].scatter(X.to_numpy()[Y_predicted == 0][:, 0], X.to_numpy()[Y_predicted == 0][:, 1], color='red')
    axis[1].set_title('Predicted data')
    plt.show()
    
    print(model.get_coef())
    
if __name__ == '__main__':
    main()