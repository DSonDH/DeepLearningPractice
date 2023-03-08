import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_size, stride, padding):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / filter_size ** 2

    def forward(self, inputs):
        self.inputs = inputs
        padded_inputs = np.pad(inputs, [(0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)])
        batch_size, height, width, num_channels = padded_inputs.shape
        output_height = (height - self.filter_size) // self.stride + 1
        output_width = (width - self.filter_size) // self.stride + 1
        self.outputs = np.zeros((batch_size, output_height, output_width, self.num_filters))

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    for f in range(self.num_filters):
                        receptive_field = padded_inputs[b, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, :]
                        self.outputs[b, i, j, f] = np.sum(receptive_field * self.filters[f, ...])
        return self.outputs

    def backward(self, grad_output, learning_rate):
        batch_size, height, width, num_channels = self.inputs.shape
        grad_inputs = np.zeros(self.inputs.shape)
        grad_filters = np.zeros(self.filters.shape)

        padded_inputs = np.pad(self.inputs, [(0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)])
        padded_grad_inputs = np.pad(grad_inputs, [(0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)])

        for b in range(batch_size):
            for i in range(self.outputs.shape[1]):
                for j in range(self.outputs.shape[2]):
                    for f in range(self.num_filters):
                        receptive_field = padded_inputs[b, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, :]
                        grad_filters[f, ...] += receptive_field * grad_output[b, i, j, f]
                        padded_grad_inputs[b, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, :] += self.filters[f, ...] * grad_output[b, i, j, f]

        grad_inputs = padded_grad_inputs[:, self.padding:-self.padding, self.padding:-self.padding, :]
        self.grad_filters = grad_filters
        return grad_inputs


class ReLU:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_grad):
        return output_grad * (self.input > 0)


class MaxPoolLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, height, width, num_channels = inputs.shape
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        self.outputs = np.zeros((batch_size, output_height, output_width, num_channels))

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    receptive_field = inputs[b, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size, :]
                    self.outputs[b, i, j, :] = np.amax(receptive_field, axis=(0, 1))
        return self.outputs
    
    def backward(self, grad_output, learning_rate):
        grad_inputs = np.zeros(self.inputs.shape)
        batch_size, height, width, num_channels = self.inputs.shape

        for b in range(batch_size):
            for i in range(grad_output.shape[1]):
                for j in range(grad_output.shape[2]):
                    receptive_field = self.inputs[b, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size, :]
                    mask = (receptive_field == np.amax(receptive_field, axis=(0, 1)))
                    grad_inputs[b, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size, :] += mask * grad_output[b, i, j, :]
        return grad_inputs

        
class FlattenLayer:
    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape((self.input_shape[0], -1))
    
    def backward(self, grad_output, learning_rate):
        return np.reshape(grad_output, self.inputs_shape)

class DenseLayer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons

    def forward(self, input):
        self.input = input
        num_inputs = input.shape[1]
        self.weights = np.random.randn(num_inputs, self.num_neurons) / num_inputs
        self.bias = np.zeros((1, self.num_neurons))
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    def backward(self, d_output, learning_rate):
        d_input = np.dot(d_output, self.weights.T)
        d_weights = np.dot(self.input.T, d_output)
        d_bias = np.sum(d_output, axis=0, keepdims=True)
        self.weights -= learning_rate * d_weights / self.input.shape[0]
        self.bias -= learning_rate * d_bias / self.input.shape[0]
        return d_input

class SoftmaxLayer:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs)
        self.probs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return self.probs

    def backward(self, grad_output, learning_rate):
        num_samples = grad_output.shape[0]
        jacobian_matrix = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                if i == j:
                    jacobian_matrix[i, j] = self.probs[i] * (1.0 - self.probs[i])
                else:
                    jacobian_matrix[i, j] = -self.probs[i] * self.probs[j]
        grad_input = np.dot(grad_output, jacobian_matrix)
        return grad_input


class CNN:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)

    def train(self, X, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            loss = 0
            
            input = X
            for layer in self.layers:
                input = layer.forward(input)

            loss += np.sum((input - y[i])**2)
            d_output = 2 * (input - y[i])
            for layer in reversed(self.layers):
                d_output = layer.backward(d_output, learning_rate)
            print("Epoch %d loss: %.4f" % (epoch+1, loss/X.shape[0]))

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            input = X[i]
            for layer in self.layers:
                input = layer.forward(input)
                prediction = np.argmax(input)
                predictions.append(prediction)
                return np.array(predictions)

if __name__ == "__main__":
    # generate random training data
    X_train = np.random.randn(100, 28, 28, 1)
    y_train = np.zeros((100, 10))
    for i in range(100):
        label = np.random.randint(10)
        y_train[i, label] = 1

    # create the CNN model
    cnn = CNN()
    cnn.add(ConvLayer(8, 3, 1, 0))
    cnn.add(ReLU())
    cnn.add(MaxPoolLayer(2, 2))
    cnn.add(FlattenLayer())
    cnn.add(DenseLayer(10))
    cnn.add(SoftmaxLayer())

    # train the model
    cnn.train(X_train, y_train, num_epochs=10, learning_rate=0.1)

    # generate random test data
    X_test = np.random.randn(10, 28, 28, 1)

    # predict the test data
    predictions = cnn.predict(X_test)

    print("Predictions:", predictions)
