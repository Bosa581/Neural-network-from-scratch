#import numpy
import random
import io

from config import LEARNING_RATE
from formulas import sig, inv_sig

curr_node_id = 0

class Layer:
    def __init__(self, num_nodes, input_vals, layer_num):
        self.num_nodes = num_nodes                     # number of neurons in this layer
        self.input_vals = input_vals                   # incoming inputs from previous layer
        self.layer_num = layer_num                     # identifier for debugging or ordering

        # weight[j][i] is the weight from input neuron i to neuron j in this layer
        self.weight = [[(random.random() * 2) - 1 for _ in range(len(input_vals))]
               for _ in range(num_nodes)]
        self.weight_delta = [[0 for _ in range(len(input_vals))] for _ in range(num_nodes)]  # unused but kept for compatibility

        self.layer_net = [0 for _ in range(num_nodes)] # pre-activation (Σ w_ij * input_i + bias_j)
        self.layer_out = [0 for _ in range(num_nodes)] # post-activation output

        # bias[j] is the bias term for neuron j, initialized randomly in [-1, 1]
        self.bias = [(random.random() * 2) - 1 for _ in range(num_nodes)]

        # holds the delta values computed during backpropagation
        self.deltas = [0 for _ in range(num_nodes)]

    def eval(self):
        # compute weighted input and apply activation function for each neuron
        for j in range(self.num_nodes):
            net = 0
            for i in range(len(self.input_vals)):
                net += self.weight[j][i] * self.input_vals[i]  # accumulate weighted input values
            net += self.bias[j]                                # add bias term
            self.layer_net[j] = net                            # store net input
            self.layer_out[j] = sig(net)                       # apply sigmoid activation
        return self.layer_out

    def backprop(self, target_or_next):
        # compute delta terms for this layer, depending on whether this is output or hidden
        deltas = [0 for _ in range(self.num_nodes)]

        # output layer: target data provided directly
        if isinstance(target_or_next, list):
            for j in range(self.num_nodes):
                # error gradient is (output - target) for stable gradient descent direction
                # RIGHT: negative of the gradient, so that w += ... goes downhill
                err_grad = target_or_next[j] - self.layer_out[j]
                act_grad = inv_sig(self.layer_net[j])                # derivative of sigmoid(net)
                deltas[j] = err_grad * act_grad                     # final delta for output neuron

        # hidden layer: propagate error using next layer's deltas and weights
        else:
            next_layer = target_or_next
            for j in range(self.num_nodes):
                sum_term = 0
                for k in range(next_layer.num_nodes):
                    # contribution from each neuron in next layer scaled by its weight
                    sum_term += next_layer.deltas[k] * next_layer.weight[k][j]
                deltas[j] = inv_sig(self.layer_net[j]) * sum_term  # apply activation derivative

        self.deltas = deltas  # store deltas so earlier layers can use them

        # update each weight and bias using computed deltas
        for j in range(self.num_nodes):
            for i in range(len(self.input_vals)):
                # gradient descent: move weights in direction that reduces error
                self.weight[j][i] += LEARNING_RATE * deltas[j] * self.input_vals[i]
        for j in range(self.num_nodes):
            self.bias[j] += LEARNING_RATE * deltas[j]  # update bias for neuron j


class cfile(io.TextIOWrapper):
    def __init__(self, name, mode='r'):
        self._file = open(name, mode)                                   # open raw file handle
        super().__init__(self._file.buffer, encoding='utf-8')           # wrap for text writing

    def w(self, string):
        self.write(str(string) + '\n')                                  # write followed by newline
        self.flush()                                                    # push to disk immediately

    def close(self):
        super().close()                                                 # close wrapper
        self._file.close()                                              # close underlying file
