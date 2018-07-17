import pyneurons
import mnist_read
import numpy


def mnist_test_network(neural_network, input_arrays, output_arrays):
    correct_guesses = 0
    wrong_guesses = 0

    amount_of_arrays = len(input_arrays)
    for array_i in range(amount_of_arrays):
        input_array = input_arrays[array_i]
        output_array = output_arrays[array_i]

        snapshot = neural_network.snapshot(input_array)
        output_layer = snapshot[-1]

        print(output_layer)
        print(output_array)

        if numpy.argmax(output_array) == numpy.argmax(output_layer):
            correct_guesses += 1
        else:
            wrong_guesses += 1

    print("Test > n: " + str(amount_of_arrays)
          + ", Accuracy: " + str(round(correct_guesses / (correct_guesses + wrong_guesses), 4))
          + ", Correct Guesses: " + str(correct_guesses)
          + ", Wrong Guesses: " + str(wrong_guesses))


classifier = pyneurons.NeuralNetwork()
classifier.add_layer(784)
classifier.add_layer(16)
classifier.add_layer(16)
classifier.add_layer(10)
classifier.assemble()

train_input_arrays = mnist_read.train_images
train_output_arrays = pyneurons.Tools.mnist_labels_to_arrays(mnist_read.train_labels)

pyneurons.Tools.train_network(classifier, train_input_arrays[:30000], train_output_arrays[:30000], 500, 0.01)

test_input_arrays = mnist_read.test_images
test_output_arrays = pyneurons.Tools.mnist_labels_to_arrays(mnist_read.test_labels)

mnist_test_network(classifier, test_input_arrays, test_output_arrays)
