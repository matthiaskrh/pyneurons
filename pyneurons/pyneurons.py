import numpy
import random
import time


class Tools:
    @staticmethod
    def train_network(neural_network, input_arrays, output_arrays, training_group_size, learning_rate):
        """
        :param neural_network: (NeuralNetwork) The network you want to train.
        :param input_arrays: (List) Entire dataset you want to train with.
        :param output_arrays: (List) Entire dataset you want to train with.
        :param training_group_size: (int) How many datasets to train at one time.
        :param learning_rate: (int) How much the network adjusts to learn.
        """

        # Shuffling arrays
        numpy.random.shuffle(input_arrays)
        numpy.random.shuffle(output_arrays)

        # Separating data into groups
        if len(input_arrays) % training_group_size == 0:
            amount_of_groups = (len(input_arrays) // training_group_size)
        else:
            amount_of_groups = (len(input_arrays) // training_group_size) + 1

        for group_i in range(amount_of_groups):
            input_training_group = []
            output_training_group = []

            if group_i == amount_of_groups - 1 and len(input_arrays) % training_group_size != 0:
                remainder = len(input_arrays) % training_group_size
                for last_array_i in range(len(input_arrays) - remainder, len(input_arrays)):
                    input_training_group.append(input_arrays[last_array_i])
                    output_training_group.append(output_arrays[last_array_i])
            else:
                for array_i in range(group_i * training_group_size, (group_i + 1) * training_group_size):
                    input_training_group.append(input_arrays[array_i])
                    output_training_group.append(output_arrays[array_i])

            # Training network
            neural_network.train(input_training_group, output_training_group, learning_rate)

            # Print progress
            print("Training Progress: " + str(round((group_i + 1) / amount_of_groups * 100, 4)) + "%")

    @staticmethod
    def mnist_labels_to_arrays(output_labels):
        converted_arrays = []
        for label in output_labels:
            converted_array = []
            for output_i in range(10):
                if output_i == label:
                    converted_array.append(1)
                else:
                    converted_array.append(0)
            converted_arrays.append(converted_array)

        return converted_arrays


class NeuralNetwork:

    # Initialization
    def __init__(self):

        # Initializing Vars
        self.layers = []  # Array of layers, X is layers, Y is layers

    def add_layer(self, neuron_amount, layer_type="default", activation_type="sigmoid"):
        """
        :param neuron_amount: (int) Amount of neurons in the added layer.
        :param layer_type: (string) Type of layer.
        :param activation_type: (string) Type of activation function for all neurons in layer.
        """

        new_layer = []
        for neuron_i in range(neuron_amount):
            new_neuron = Neuron(layer_type, activation_type)
            new_layer.append(new_neuron)

        self.layers.append(new_layer)

    def assemble(self, randomize_type="uniform"):
        """
        Connects all layers to each other and randomize their weights and biases.
        :param randomize_type: (string) Random function that is applied to the weights and biases.
        """

        if len(self.layers) < 2:
            raise RuntimeError("Can't assemble. Need at least an input layer and output layer.")

        # Connecting and randomizing weights and randomizing biases
        for layer_i in range(1, len(self.layers)):
            layer = self.layers[layer_i]
            previous_layer = self.layers[layer_i - 1]

            for neuron_i in range(len(layer)):
                neuron = layer[neuron_i]
                neuron.input_weights = numpy.ones(len(previous_layer))
                neuron.bias = _Utility.random_function(randomize_type)

                for weights_i in range(neuron.input_weights.size):
                    neuron.input_weights[weights_i] = _Utility.random_function(randomize_type)

    # Training
    def train(self, input_array_set, output_array_set, learning_rate):
        """
        :param input_array_set: (list) Input data that you want to train with.
        :param output_array_set: (list) Output data that you want to train with.
        :param learning_rate: (int) How much the network adjusts to learn.
        """

        # Initializing Arrays
        network_weights = []  # X is layer, Y is neuron, Z is numpy array of weights
        network_biases = []  # X is layer, Y is numpy array of biases

        for layer_i in range(len(self.layers)):
            layer = self.layers[layer_i]

            network_biases.append(numpy.zeros(len(layer)))

            layer_weights = []
            for neuron_i in range(len(layer)):
                neuron = layer[neuron_i]
                layer_weights.append(numpy.zeros(neuron.input_weights.size))
            network_weights.append(layer_weights)

        # Getting average cost
        average_cost = 0

        # Calculating weight and bias adjustments
        amount_of_arrays = len(input_array_set)
        for group_i in range(amount_of_arrays):
            network_snapshot = self.snapshot(input_array_set[group_i])
            d_out_array = numpy.subtract(network_snapshot[-1], output_array_set[group_i])

            average_cost += self.cost(network_snapshot[-1], output_array_set[group_i]) / amount_of_arrays

            # Iterating layers from output to layers[1]
            for layer_i in range(len(self.layers) - 1, 0, -1):
                layer = self.layers[layer_i]
                next_layer = self.layers[layer_i - 1]
                next_d_out_array = numpy.zeros(len(next_layer))

                for neuron_i in range(len(layer)):
                    neuron = layer[neuron_i]

                    previous_layer_values = network_snapshot[layer_i - 1]
                    out_value = network_snapshot[layer_i][neuron_i]

                    # Getting derivatives
                    d_out = d_out_array[neuron_i]
                    d_activation = _Utility.activation_derivative(out_value, neuron.activation_type)
                    d_rule = d_out * d_activation

                    # Finding better weights and biases in current layer
                    w_gradients = numpy.multiply(previous_layer_values, d_rule)
                    w_gradients = numpy.multiply(w_gradients, learning_rate)
                    new_weights = numpy.subtract(neuron.input_weights, w_gradients)

                    b_gradient = d_activation * d_out
                    b_gradient *= learning_rate
                    new_bias = neuron.bias - b_gradient

                    new_weights = numpy.divide(new_weights, amount_of_arrays)
                    new_bias = new_bias / amount_of_arrays

                    network_weights[layer_i][neuron_i] = numpy.add(network_weights[layer_i][neuron_i], new_weights)
                    network_biases[layer_i][neuron_i] += new_bias

                    # Calculating next d_out_array
                    sub_d_out_array = numpy.multiply(neuron.input_weights, d_rule)
                    next_d_out_array = numpy.add(next_d_out_array, sub_d_out_array)

                d_out_array = next_d_out_array

        # Applying new weights
        for layer_i in range(1, len(self.layers)):
            layer = self.layers[layer_i]

            for neuron_i in range(len(layer)):
                neuron = layer[neuron_i]

                neuron.input_weights = network_weights[layer_i][neuron_i]
                neuron.bias = network_biases[layer_i][neuron_i]

        # Printing Diagnostics
        print("Average Cost:", round(average_cost, 4))

    def snapshot(self, input_array):
        """
        Takes a snapshot of neuron values for a certain input.
        :param input_array: (list) of floats for the input layer.
        :return: (list) X is layer, Y is numpy array corresponding to the strength of each neuron
        """

        # Checking if input dataset is compatible
        if len(input_array) != len(self.layers[0]):
            raise RuntimeError(
                "Input dataset is not 1 to 1 with input layer. Need dataset to be the same length as input layer")

        # Setting up initial values for feed forward
        # X is layer, Y is numpy array corresponding to the strength of each neuron
        test_snapshot = [numpy.array(list(input_array))]

        # Feeding forward
        for layer_i in range(1, len(self.layers)):
            layer = self.layers[layer_i]

            # Getting previous layer values
            previous_layer_values = test_snapshot[-1]

            # Creating array for current layer values
            layer_values = numpy.empty(len(layer))

            # Getting values for all neurons in layer
            for neuron_i in range(len(layer)):
                neuron = layer[neuron_i]
                layer_values[neuron_i] = neuron.feed_forward_value(previous_layer_values)

            test_snapshot.append(layer_values)

        # Returning output dataset
        return test_snapshot

    @staticmethod
    def cost(out_array, target_array):
        """
        :param out_array: (list) of float values for the actual output layer.
        :param target_array: (list) of float values for the desired output layer.
        :returns: (float) total cost of the network.
        """

        # Calculating cost with (goal - current) ^ 2 function
        dataset_difference = numpy.subtract(list(target_array), list(out_array))
        dataset_product = numpy.multiply(dataset_difference, dataset_difference)
        return numpy.sum(dataset_product)

    # Debug
    def print_structure_info(self):
        """
        Prints all information about layers in the network.
        """

        for layer_i in range(len(self.layers)):
            layer = self.layers[layer_i]

            for neuron_i in range(len(layer)):
                neuron = layer[neuron_i]

                print(
                    "Layer: " + str(layer_i) +
                    " > Neuron: " + str(neuron_i) +
                    ", Bias: " + str(neuron.bias)),

                for weight_i in range(neuron.output_weights.size):
                    weight = neuron.output_weights[weight_i]
                    print(
                        " > Connection: " + str(weight_i) +
                        ", Weight: " + str(weight))


class Neuron:
    def __init__(self, neuron_type, activation_type):

        # Initializing Vars
        self.neuron_type = neuron_type
        self.activation_type = activation_type
        self.bias = 0
        self.input_weights = numpy.empty(0)

    def feed_forward_value(self, previous_layer_values):
        """
        :param previous_layer_values: (nparray)
        :return: (float) Value of this neuron.
        """

        weighted_values = numpy.multiply(self.input_weights, previous_layer_values)
        weighted_sum = numpy.sum(weighted_values)
        weighted_sum += self.bias

        return _Utility.activation_function(weighted_sum, self.activation_type)

class _Utility:
    @staticmethod
    def activation_function(x, activation_type):
        if activation_type == "relu":
            return numpy.maximum(x, 0)

        if activation_type == "sigmoid":
            return 1 / (1 + numpy.exp(-x))

        if activation_type == "tanh":
            return numpy.tanh(x)

    @staticmethod
    def activation_derivative(x, activation_type):
        if activation_type == "relu":
            return x > 0

        if activation_type == "sigmoid":
            return x * (1 - x)

        if activation_type == "tanh":
            return 1 - (x * x)

    @staticmethod
    def random_function(randomize_type):

        if randomize_type == "uniform":
            return random.uniform(-1, 1)


