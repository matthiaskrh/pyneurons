# Opening files
test_images_data = open("t10k-images.idx3-ubyte", "rb").read()
test_labels_data = open("t10k-labels.idx1-ubyte", "rb").read()
train_images_data = open("train-images.idx3-ubyte", "rb").read()
train_labels_data = open("train-labels.idx1-ubyte", "rb").read()

test_labels = test_labels_data[8:]
train_labels = train_labels_data[8:]

test_images_data = test_images_data[16:]
train_images_data = train_images_data[16:]

test_images = []
for x in range(len(test_images_data) // 784):
    test_images.append(test_images_data[x * 784: (x + 1) * 784])

train_images = []
for x in range(len(train_images_data) // 784):
    train_images.append(train_images_data[x * 784: (x + 1) * 784])
