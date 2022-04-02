import random
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Node:
    def __init__(self, parent_nodes=None, is_io=False):
        if parent_nodes is None:
            parent_nodes = []

        self.is_io = is_io
        self.val = 0

        if is_io:
            self.weight = random.randint(0, 1)
            self.parents = parent_nodes
        else:
            self.parents = parent_nodes
            self.bias = random.randint(0, 1)
            self.weight = random.randint(0, 1)

    def eval_value(self):
        # get value from other parent nodes:
        output = 0
        for i in self.parents:
            output += i.weight * i.number

        output -= self.bias

        self.val = sigmoid(output)
        return output

    def assign_value(self, val):
        if not self.is_io:
            raise ValueError("A normal node cannot be 'assigned' a value.")
        else:
            self.val = val


class MultilayerPerceptronNetwork:
    def __init__(self, layers):
        # create network:
        output_network = []
        counter = 0

        for i in layers:
            counter_list = []
            for j in range(i):
                # first and last index is always input nodes:
                if counter == 0:
                    counter_list.append(Node(is_io=True))
                # when it's not first one:
                elif counter != len(layers) - 1:
                    counter_list.append(Node(parent_nodes=output_network[layers.index(i) - 1]))
                else:
                    counter_list.append(Node(parent_nodes=output_network[layers.index(i) - 1], is_io=True))

            output_network.append(counter_list)
            counter += 1

        self.network = output_network
        self.layers = layers