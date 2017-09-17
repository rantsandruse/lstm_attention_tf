from sklearn.cross_validation import StratifiedShuffleSplit


def train_test_split_shuffle(target, features, test_size = 0.1):
    sss = StratifiedShuffleSplit(target, 1, test_size = test_size, random_state=0)
    for train_index, test_index in sss:
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]
        y_test = y_test.values
        y_train = y_train.values

    return X_train, y_train, X_test, y_test


