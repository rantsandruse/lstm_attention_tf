import numpy as np
import random

class BatchGenerator:
    def __init__(self, X_train, y_train, X_test, y_test, batchSize, shuffle=True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_train_offset = 0
        self.X_test_offset = 0
        self.shuffle = shuffle
        self.batchSize = batchSize

    def shuffleIfTrue(self):
        if self.shuffle:
            arr = np.arange(len(self.X_train))
            np.random.shuffle(arr)
            self.X_train = self.X_train[arr]
            self.y_train = self.y_train[arr]


    def nextTrainBatch(self):
        start = self.X_train_offset
        end = self.X_train_offset + self.batchSize

        self.X_train_offset = end

        # handle wrap around    ,
        if end > len(self.X_train):
            spillover = end - len(self.X_train)
            self.X_train_offset = spillover
            X = np.concatenate((self.X_train[start:], self.X_train[:spillover]), axis = 0)
            y = np.transpose([ np.concatenate((self.y_train[start:], self.y_train[:spillover]), axis = 0),
                               1 - np.concatenate((self.y_train[start:], self.y_train[:spillover]), axis = 0)])

            self.X_train_offset = 0
            self.shuffleIfTrue()

        else:
            X = self.X_train[start:end]
            y = np.transpose([self.y_train[start:end], 1- self.y_train[start:end]])

        X = X.astype(np.int32, copy = False)
        y = y.astype(np.float32, copy = False)

        return X,y

    def nextTestBatch(self):
        start = self.X_test_offset
        end = self.X_test_offset + self.batchSize
        self.X_test_offset = end

        # handle wrap around    ,
        if end > len(self.X_test):
            spillover = end - len(self.X_test)
            self.X_test_offset = spillover
            X = np.concatenate((self.X_test[start:], self.X_test[:spillover]), axis=0)
            y = np.transpose([np.concatenate((self.y_test[start:], self.y_test[:spillover]), axis=0),
                              1 - np.concatenate((self.y_test[start:], self.y_test[:spillover]), axis=0)])

            self.X_test_offset = 0

            self.shuffleIfTrue()

        else:
            X = self.X_test[start:end]
            y = np.transpose([self.y_test[start:end], 1 - self.y_test[start:end]])

        X = X.astype(np.int32, copy=False)
        y = y.astype(np.float32, copy=False)

        return X, y



