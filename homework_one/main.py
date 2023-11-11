import copy
import pickle, gzip, numpy as np


class NeuralNetwork:
    def __init__(self, train, labels, alpha = 0.01, batch_size = 128, epochs = 100):
        self.train = copy.deepcopy(train) # train_shape = (60000, 784)
        self.labels = copy.deepcopy(labels) # labels_shape = (60000,)
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights = np.random.randn(784, 10) # weights_shape = (784, 10)
        self.bias = np.random.randn(10) # bias_shape = (10,)

    def softmax(self, current_batch):
        max_on_each_row = np.max(current_batch, axis=1, keepdims=True) # for each row we get the max element
        current_batch_after_removing_max = current_batch - max_on_each_row # we remove the max element from each row
        exp_each_element = np.exp(current_batch_after_removing_max) # we apply the exponential function to each element
        sum_on_each_row = np.sum(exp_each_element, axis=1, keepdims=True) # we sum the elements on each row
        return exp_each_element / sum_on_each_row # we divide each element by the sum of the elements on the same row

    def shuffle(self):
        permutation = np.random.permutation(self.train.shape[0])
        self.train = self.train[permutation]
        self.labels = self.labels[permutation]

    def forward(self, current_batch):
        # shape = (128, 784) * shape = (784, 10) + shape = (10,) => shape (128, 10)
        return self.softmax(current_batch.dot(self.weights) + self.bias) 
    
    def loss(self, predictions, labels): # MSE
        m = predictions.shape[0]
        calculated_prediction_for_each_label = np.zeros(m)
        for i in range(m):
            calculated_prediction_for_each_label[i] = predictions[i][labels[i]]
        error = np.mean((calculated_prediction_for_each_label - 1) ** 2)
        return error # 1/m * sum((y - y_hat) ** 2)
    

    def train_step(self):
        for _ in range(self.epochs):
            self.shuffle()

            for idx in range(0, self.train.shape[0], self.batch_size):
                self.batch_train = self.train[idx:idx + self.batch_size] # batch_train_shape = (128, 784)
                self.batch_labels = self.labels[idx:idx + self.batch_size] # batch_labels_shape = (128,)

                predictions = self.forward(self.batch_train) # predictions_shape = (128, 10)

                m = len(self.batch_train)
                dz = predictions

                for i in range(m): # dz = predictions - labels
                    dz[i] -= [1 if j == self.batch_labels[i] else 0 for j in range(10)]

                dw = 1 / m * np.dot(self.batch_train.T, dz) # dw = 1/m * (X.T * dz)
                db = 1 / m * np.sum(dz, axis=0)

                self.weights -= self.alpha * dw
                self.bias -= self.alpha * db


    def test(self, train, labels):
        predictions = self.forward(train)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == labels)
        print("Accuracy: ", accuracy)


with gzip.open("mnist.pkl.gz", "rb") as fd:
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = pickle.load(fd, encoding="latin")

train = np.concatenate((train_x, test_x))
labels = np.concatenate((train_y, test_y))

nn = NeuralNetwork(train, labels)

nn.train_step()

print("Training set ", end="")
nn.test(train, labels)

print("Validation set ", end="")
nn.test(valid_x, valid_y)
