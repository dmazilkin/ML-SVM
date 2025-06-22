import numpy as np
import pandas as pd
from typing import Tuple

from src.svm import MySVM

N = 500

def create_data() -> Tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    
    X = pd.DataFrame(np.zeros((N, 2)) + 10 * np.random.rand(N, 2))
    Y = (X.loc[:, 0] > X.loc[:, 1] - 2).astype(int)
    X = pd.concat([X, X.loc[:, 1]**2, X.loc[:, 1]**3], axis=1)
    
    return X, Y

def main():
    X, Y = create_data()
    model = MySVM()
    model.fit(X, Y)
    
if __name__ == '__main__':
    main()