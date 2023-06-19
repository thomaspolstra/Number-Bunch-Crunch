from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np 
from datetime import datetime, timedelta


class VolatilityRandomForest:
    def __init__(self, n_estimators: int = 100, max_depth: int | None = None):
        """
        :param n_estimators: number of decision trees in the forest
        :param max_depth: max_depth of decision trees used in random forest
        """
        self.rf = RandomForestRegressor(n_estimators=n_estimators,
                                        max_depth=max_depth)
        
    def fit(self, X, y) -> RandomForestRegressor:
        """
        fits the RandomForestRegressor model. 
        :param X: MatrixLike
        :param y: MatrixLike
        """
        return self.rf.fit(X,y)
    

    def predict(self, X: np.ndarray | pd.core.series.Series, 
                n_windows: int = 1) -> any:
        """
        predicts the volatility given an array and returns the predictions.
        :param X: array-like series of data to predict.
        :param n_windows: the number of windows used to generate predictions.
        :return: array of predictions.
        """
        X_preds = X[:n_windows].copy()

        new_size = len(X) - n_windows
        
        preds = np.zeros(len(X))

        for i in range(n_windows):
            preds[i] = self.rf.predict(X_preds[i].reshape(1,-1))
        
        for i in range(0,new_size):
            past_row = X_preds[-1][:-1]
            new_row = np.insert(past_row, 0, preds[n_windows+i-1]).reshape(1,-1)
            X_preds = np.concatenate((X_preds, new_row), axis=0)
            preds[n_windows + i] = self.rf.predict(X_preds[-1].reshape(1,-1))

        return preds