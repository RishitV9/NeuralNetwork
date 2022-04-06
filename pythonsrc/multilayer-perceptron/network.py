import random
import math
import time


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Node:
    def __init__(self, parent_nodes=None, is_io=False, bias=0.0, weight_headers=None):

        """

        :param parent_nodes: list
        :param is_io: bool
        :param bias: float
        :param weight_headers: list
        """

        self.is_io = is_io
        self.val = 0.0

        if is_io:
            self.parents = parent_nodes
            self.bias = 0.0
        else:
            self.parents = parent_nodes
            if bias == 0.0:
                self.bias = random.random()
            else:
                self.bias = bias


        self.weights = {}
        if weight_headers is None:
            parents = []

            if parent_nodes is not None:
                parents += parent_nodes

            for i in parents:
                self.weights[i] = random.random()
        else:
            parents = []

            if parent_nodes is not None:
                parents += parent_nodes

            counter = 0
            for i in parents:
                try:
                    self.weights[i] = weight_headers[counter]
                except:
                    pass

                counter += 1

    def eval_value(self):
        # get value from other parent nodes:
        output = 0.0
        for i in self.parents:
            output += self.weights[i] * i.val

        output -= self.bias
        self.val = sigmoid(output)
        return output

    def assign_value(self, val):
        if not self.is_io:
            raise ValueError("A normal node cannot be 'assigned' a value.")
        else:
            self.val = val


class MultilayerPerceptronNetwork:
    def __init__(self, layers=None, location=""):
        
        if location == "":
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
            self.computation_duration = None
        else:
            if layers is None:
                raise ValueError("Layers information is not provided.")

            network = []
            counter = 0
            with open(f'{location}export.txt', 'r') as file:
                data = file.readlines()
                for i in layer:
                    each_layer = []
                    for j in range(i):
                        # append the node to each_layer:
                        if "True" in data:
                            is_io = True
                        else:
                            is_io = False
                        
                        bias = float(data[counter].split()[0]
                        weights = {}
                        other_data = data[counter].split().pop(0).pop(len(data[counter] - 1))

                        # for i in other_data:
                            # 

                    network.append(each_layer)

    def run(self, args):
        if len(self.network[0]) != len(args):
            raise ValueError("there are too many or too less arguments for input nodes given.")

        counter = 0
        counter2 = 0

        start = time.time()
        for i in self.network:
            for j in i:
                if counter == 0:
                    j.assign_value(args[counter2])
                else:
                    j.eval_value()
                counter2 += 1
            counter += 1
        end = time.time()

        self.computation_duration = end - start

        return self.network[len(self.network) - 1][0].eval_value()

    def export_network(self, location=""):
        with open(f"{location}export.txt", "w") as file:
            data = ''

            for i in self.network:
                for j in i:
                    weights = ""

                    for c in j.weights:
                        weights += str(j.weights[c]) + " "

                    data += f"{j.bias} {weights}{j.is_io}\n"

            file.write(data)
            file.close()
