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
        else:
            self.parents = parent_nodes
            self.bias = random.randint(0, 1)
            self.weight = random.randint(0, 1)

    def eval_value(self, args):
        # get value from other parent nodes:
        output = 0
        for i in self.parents:
            output += i.weight * i.number

        output -= self.bias

        self.val = sigmoid(output)
        return output

