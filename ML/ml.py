import numpy as np
from sklearn.ensemble import RandomForestRegressor
import datetime

class LikePredictor():
    def __init__(self, data):
        self.data = np.array(data)

        # Randomize row order for training examples
        np.random.shuffle(self.data)

        # Number of training examples
        self.m = self.data.shape[0]
        # Number of features
        self.n = self.data.shape[1] - 1

        X = self.data[:, :self.n]
        y = self.data[:, self.n]

        self.regressor = RandomForestRegressor()
        self.regressor.fit(X, y)

    def predict(self, img_vector):
        now = datetime.datetime.now()
        input_vector = np.hstack(([img_vector], [[int(now.strftime("%s")) * 1000]]))
        input_vector = np.hstack((input_vector, [[now.hour]]))
        return self.regressor.predict(input_vector)[0]