import random
import time

random.seed(1)
import numpy as np


class Linear:
    def __init__(self, input, out):
        kw = 10000
        k = np.sqrt(1 / input) * kw
        self.w = np.random.randint(-k, k, size=(out, input)) / kw
        self.b = np.random.randint(-k, k, size=(out)) / kw

    def __call__(self, *args, **kwargs):
        x = args[0]
        self.x = x
        return np.inner(x, self.w) + self.b

    def get_parameter(self):
        return self.w, self.b


class Relu:
    def __call__(self, *args, **kwargs):
        x = np.array(args[0])
        return np.maximum(0, x)


class SofeMax:
    def __call__(self, *args, **kwargs):
        x = np.array(args[0])
        return np.exp(x) / np.sum(np.exp(x))


class Adam:
    def __init__(self, linear: Linear):
        self.w = linear.w
        self.b = linear.b

    def send(self, x, loss):
        dw = 2 * x * loss
        db = 2 * loss
        length = 0.01
        self.w += length * dw
        self.b += length * db


y = np.zeros(10)
y[2] = 1
x = np.array([random.random() for i in range(10)])
linear = Linear(10, 10)
adam = Adam(linear)
while True:
    xq = linear(x)
    xq = Relu()(xq)
    xq = SofeMax()(xq)
    loss = np.mean((y - xq) ** 2)
    print(loss)
    time.sleep(0.5)
    adam.send(x, loss)
    x = xq
    time.sleep(0.1)
