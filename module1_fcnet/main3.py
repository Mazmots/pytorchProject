#%% 演示loss.backward() 和optimizer.step()的用法：
import torch

# 初始化x
# requires_grad=True让 backward 可以追踪这个参数并且计算它的梯度
x = torch.tensor([1., 2.], requires_grad=True)

# 模拟网络运算
y = 100 * x

# 损失
loss = y.sum()

print(f'x:{x}')
print(f'y:{y}')
print(f'loss:{loss}')

print(f'反向传播前，参数的梯度是:{x.grad}')

loss.backward()

print(f'反向传播后，参数的梯度是:{x.grad}')

# 优化器
optim = torch.optim.SGD([x], lr=0.001)

print(f'更新参数前,x为：{x}')
optim.step()
print(f'更新参数后,x为：{x}')

# 再进行一次网络运算
y = 100 * x

# 定义损失
loss = y.sum()

# 不进行optimizer.zero_grad()
optim.zero_grad()
loss.backward()  # 计算梯度grad, 更新 x*grad
print("不进行optimizer.zero_grad(), 参数的梯度为: ", x.grad)




#%%
"""
在该示例代码中，我们定义了一个名为NeuralNetwork的类，该类具有以下几个方法：



__init__()方法用于初始化权重和偏置；

sigmoid()方法用于对输入进行sigmoid激活；

forward()方法用于进行前向传播，计算隐藏层和输出层的输出；

sigmoid_derivative()方法用于计算sigmoid函数的导数；

backward()方法用于进行反向传播，更新权重和偏置；

train()方法用于训练神经网络，采用指定的训练集和期望输出，以及指定的训练次数；

predict()方法用于使用已训练好的模型对指定的输入进行预测，输出预测结果。


示例代码中的神经网络采用了单隐层结构，可以根据需要进行修改。

"""
import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        # sigmoid激活函数
        return 1 / (1 + np.exp(-x))

    def forward(self, input):
        # 前向传播
        self.hidden = self.sigmoid(np.dot(input, self.weights1) + self.bias1)
        self.output = self.sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)

    def sigmoid_derivative(self, x):
        # sigmoid函数的导数
        return x * (1 - x)

    def backward(self, input, output, expected_output):
        # 反向传播
        error = expected_output - output
        d_output = error * self.sigmoid_derivative(output)

        error_hidden = d_output.dot(self.weights2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.hidden)

        # 更新权重和偏置
        self.weights2 += self.hidden.T.dot(d_output)
        self.bias2 += np.sum(d_output, axis=0, keepdims=True)
        self.weights1 += input.T.dot(d_hidden)
        self.bias1 += np.sum(d_hidden, axis=0, keepdims=True)

    def train(self, input, expected_output, epochs):
        for i in range(epochs):
            self.forward(input)
            self.backward(input, self.output, expected_output)

    def predict(self, input):
        self.forward(input)
        return self.output



#%%

import numpy as np


# Define the sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Define the derivative of the sigmoid activation function
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


# Define a function for initializing the weights and biases
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_outputs):
    # Initialize a list for holding the weights and biases of each layer
    weights = []
    biases = []

    # Initialize the weights and biases of the input layer to be zero
    weights.append(np.zeros((num_inputs, num_nodes_hidden)))
    biases.append(np.zeros((1, num_nodes_hidden)))

    # Initialize the weights and biases of each hidden layer using a normal distribution
    for i in range(num_hidden_layers):
        weights.append(np.random.randn(num_nodes_hidden, num_nodes_hidden))
        biases.append(np.random.randn(1, num_nodes_hidden))

    # Initialize the weights and biases of the output layer using a normal distribution
    weights.append(np.random.randn(num_nodes_hidden, num_outputs))
    biases.append(np.random.randn(1, num_outputs))

    return weights, biases


# Define a function for performing forward propagation
def forward_propagation(inputs, weights, biases):
    # Initialize the activations of the input layer
    activations = [inputs]

    # Perform forward propagation through each layer of the network
    for i in range(len(weights)):
        # Calculate the dot product of the activations and weights for the current layer
        z = np.dot(activations[i], weights[i]) + biases[i]

        # Apply the sigmoid activation function to the dot product
        activation = sigmoid(z)

        # Add the activation to the list of activations
        activations.append(activation)

    return activations


# Define a function for performing backward propagation
def backward_propagation(inputs, outputs, weights, biases, activations):
    # Initialize a list for holding the errors of each layer
    errors = [None] * len(weights)

    # Calculate the error of the output layer
    errors[-1] = (activations[-1] - outputs) * sigmoid_derivative(np.dot(activations[-2], weights[-1]) + biases[-1])

    # Perform backward propagation through each layer of the network
    for i in range(len(weights) - 2, -1, -1):
        # Calculate the error of the current layer
        errors[i] = np.dot(errors[i + 1], weights[i + 1].T) * sigmoid_derivative(
            np.dot(activations[i], weights[i]) + biases[i])

    # Initialize a list for holding the gradients of each layer
    gradients = [None] * len(weights)

    # Calculate the gradient of each layer
    for i in range(len(weights)):
        gradients[i] = np.dot(activations[i].T, errors[i])

    return gradients


# Define a function for updating the weights and biases of the network
def update_network(weights, biases, gradients, learning_rate):
    # Update the weights and biases of each layer using the calculated gradients
    for i in range(len(weights)):
        weights[i] -= learning_rate * gradients[i]
        biases[i] -= learning_rate * np.mean(gradients[i], axis=0)

    return weights, biases


# Define a function for training the network
def train_network(inputs, outputs, num_hidden_layers, num_nodes_hidden, learning_rate, num_epochs):
    # Get the number of inputs and outputs
    num_inputs = inputs.shape[1]
    num_outputs = outputs.shape[1]

    # Initialize the weights and biases of the network
    weights, biases = initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_outputs)

    # Train the network for the specified number of epochs
    for i in range(num_epochs):
        # Perform forward propagation
        activations = forward_propagation(inputs, weights, biases)

        # Perform backward propagation
        gradients = backward_propagation(inputs, outputs, weights, biases, activations)

        # Update the weights and biases
        weights, biases = update_network(weights, biases, gradients, learning_rate)

    return weights, biases


# Define some sample inputs and outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

# Train the network with the sample inputs and outputs
weights, biases = train_network(inputs, outputs, 1, 4, 0.1, 10000)

# Test the network with some new inputs
new_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
activations = forward_propagation(new_inputs, weights, biases)
print(activations[-1])


"""
该代码定义了一个全连接神经网络，并实现了几个关键函数：



sigmoid() - 激活函数，用于将该点的值映射到范围为0-1之间。

sigmoid_derivative() - 激活函数的导数，用于实现反向传播。

initialize_network() - 初始化网络的权重和偏差。

forward_propagation() - 在网络中进行前向传播。

backward_propagation() - 在网络中进行反向传播以计算梯度。

update_network() - 用梯度下降法更新网络的权重和偏差。

train_network() - 对具有指定数量的隐藏层和节点的网络进行训练。


最后，将该神经网络用于XOR逻辑运算，并输出结果。
"""