#被zzp注释掉了，没有使用这个类型


from statistics import variance
import numpy as np

class AverageEstimator:
    """使用方法：
        1. 每次可update传入不同的需要计算均值的key，并且可以利用count进行权值的配置
        2. 可以连续update多次，但需要使用1次average才生效，多次使用average是不会进行重复计算的
        3. average传入的n代表计算后n个值的平均
    """


    class AverageObject:
        def __init__(self):
            self.val = list()
            self.n = list()

        def clear(self):
            self.val.clear()
            self.n.clear()

        def update(self, values, nums):
            self.val.append(values)
            self.n.append(nums)

        def average(self, n:int = 0) -> None:
            if n == None:
                values = np.array(self.val[:])
                nums = np.array(self.n[:])
                avg = np.sum(values * nums) / sum(nums)
            else:
                values = np.array(self.val[-n:])
                nums = np.array(self.n[-n:])
                avg = np.sum(values * nums) / sum(nums)
            return avg

        def __repr__(self) -> str:
            tmpstr = self.__class__.__name__ + str(self.val) + str(self.n)
            return tmpstr

    def __init__(self):
        self.estimator = dict()
        self.avg_output = dict()

        self.ready = False

    @property
    def output(self) -> dict:
        return self.avg_output

    def clear(self):
        self.estimator.clear()
        self.avg_output.clear()
        self.ready = False

    def update(self, vars: dict, count: int = 1) -> None:
        for key, var in vars.items():
            if key not in list(self.estimator.keys()):
                self.estimator[key] = AverageEstimator.AverageObject()
                self.estimator[key].update(var, count)
            else:
                self.estimator[key].update(var, count)
        self.ready = False

    def average(self, n: int = 0) -> None:
        if self.ready == True:
            return

        # assert n >= 0 or n is None
        for key, var in self.estimator.items():
            self.avg_output[key] = self.estimator[key].average(n)
        self.ready = True

if __name__ == '__main__':
    est = AverageEstimator()
    est.update({'num1': 8}, count=0.8)
    est.update({'num1': 9})
    est.update({'num1': 10})

    est.update({'num2': 80})
    est.update({'num2': 90})
    est.update({'num2': 100})

    est.average()
    print(est.output)
    est.update({'num2': 80})
    est.update({'num2': 90})
    est.update({'num2': 100})
    est.average(2)
    print(est.output)