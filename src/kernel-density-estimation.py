import math
import numpy as np

class Kernel_d_estimator(object):
    """
    カーネル密度推定法のスクラッチ実装クラス
    """

    def __init__(self):
        pass

    def _gauss_kernel(self, xi, xj, band):
        """ガウスカーネルによる距離の計算

        :param xi np.array: x
        :param xj np.array: xi
        :param band: バンド幅
        :return:
        """
        return np.exp((-1 * (np.abs(xi - xj) ** 2)) / (2 * (band ** 2)))

    def _cal_dense(self, x):
        """確率密度の計算

        :param x　np.array: x
        :return: xにおける確率密度
        """
        norm = 1 / ((2 * math.pi * (self.band ** 2)) ** 0.5)
        return norm * np.sum(
            self._gauss_kernel(x, self.X, self.band)) / self.X_size

    def fit(self, X):
        """データの登録関数

        :param X:
        :return:
        """
        self.X = X
        self.X_size = X.shape[0]

    def predict(self, test, band):
        """指定された範囲のカーネル密度推定を行う関数

        :param test:
        :param band:
        :return:
        """
        self.band = band
        return [self._cal_dense(t) for t in test]


