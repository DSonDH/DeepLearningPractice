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
        padded_inputs = np.pad(inputs, 
                               [(0, 0), 
                                (self.padding, self.padding), 
                                (self.padding, self.padding), 
                                (0, 0)]
                              )
        batch_size, height, width, num_channels = padded_inputs.shape
        output_height = (height - self.filter_size) // self.stride + 1
        output_width = (width - self.filter_size) // self.stride + 1
        self.outputs = np.zeros((batch_size, output_height, output_width, self.num_filters))

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    for f in range(self.num_filters):
                        receptive_field = padded_inputs[
                                                        b, 
                                                        i * self.stride : i * self.stride + self.filter_size, 
                                                        j * self.stride : j * self.stride + self.filter_size, 
                                                        :
                                                       ]
                                                        
                        self.outputs[b, i, j, f] = np.sum(receptive_field * self.filters[f, ...])
        return self.outputs

    def backward(self, grad_input, learning_rate):
        batch_size, _, _, _ = self.inputs.shape
        grad_output = np.zeros(self.inputs.shape)  # 다음 레이어에 gradient 전달하기 위한 변수
        grad_filters = np.zeros(self.filters.shape)  # 현재 레이어 weight 업데이트용 변수

        padded_inputs = np.pad(self.inputs,  # 100, 28, 28, 1
                               [(0, 0), 
                                (self.padding, self.padding), 
                                (self.padding, self.padding), 
                                (0, 0)]  # pad_width : {sequence, array_like, int} 
                                         # Number of values padded to the edges of each axis.
                              )
        padded_grad_inputs = np.pad(self.inputs, 
                                    [(0, 0), 
                                     (self.padding, self.padding), 
                                     (self.padding, self.padding), 
                                     (0, 0)]
                                   )

        for b in range(batch_size):
            for i in range(self.outputs.shape[1]):
                for j in range(self.outputs.shape[2]):
                    for f in range(self.num_filters):
                        receptive_field = padded_inputs[  # 100, 28, 28, 1
                                                        b, 
                                                        j * self.stride : j * self.stride + self.filter_size, 
                                                        i * self.stride : i * self.stride + self.filter_size, 
                                                        :
                                                       ]
                        
                        grad_filters[f, ...] += (receptive_field * grad_input[b, i, j, f]).squeeze()  # 100, 26, 26, 8
                        padded_grad_inputs[
                                           b, 
                                           i * self.stride : i * self.stride + self.filter_size, 
                                           j * self.stride : j * self.stride + self.filter_size, 
                                           :
                                          ] += self.filters[f, ...] * grad_input[b, i, j, f]
        
        grad_output = padded_grad_inputs[
                                         :,
                                         self.padding : -self.padding,  # remove padding
                                         self.padding : -self.padding,
                                         :
                                        ]
        self.grad_filters = grad_filters  # 얘로 weight 업데이트 하는 코드 추가해야함.
        #TODO: weight update using learning rate 구현하고 학습 잘되는지 확인.
        
        return grad_output


class ReLU:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_grad, _):
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
                    receptive_field = inputs[
                                               b, 
                                               i * self.stride : i * self.stride + self.pool_size, 
                                               j * self.stride : j * self.stride + self.pool_size, 
                                               :
                                            ]
                                               
                    self.outputs[b, i, j, :] = np.amax(receptive_field, axis=(0, 1))
        return self.outputs
    
    def backward(self, grad_input, _):
        grad_output = np.zeros(self.inputs.shape)
        batch_size, height, width, num_channels = self.inputs.shape

        for b in range(batch_size):
            for i in range(grad_input.shape[1]):
                for j in range(grad_input.shape[2]):
                    receptive_field = self.inputs[
                                                    b, 
                                                    i * self.stride : i * self.stride + self.pool_size, 
                                                    j * self.stride : j * self.stride + self.pool_size, 
                                                    :
                                                 ]
                    mask = (receptive_field == np.amax(receptive_field, axis=(0, 1)))
                    grad_output[
                                  b, 
                                  i * self.stride : i * self.stride+self.pool_size, 
                                  j * self.stride : j * self.stride+self.pool_size, 
                                  :
                               ] += mask * grad_input[b, i, j, :]
        return grad_output

        
class FlattenLayer:
    def __init__(self) -> None:
        self.input_shape = None

    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape((self.input_shape[0], -1))
    
    def backward(self, grad_output, _):
        return np.reshape(grad_output, self.input_shape)


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

    def backward(self, grad_output, learning_rate):
        d_input = np.dot(grad_output, self.weights.T)

        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        self.weights -= learning_rate * grad_weights / self.input.shape[0]
        self.bias -= learning_rate * grad_bias / self.input.shape[0]
        
        return d_input

class SoftmaxLayer:
    def __init__(self):
        self.probs = None

    def forward(self, inputs):
        exp_inputs = np.exp(inputs)
        self.probs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return self.probs

    def backward(self, grad_output, learning_rate):
        """
        개념설명 : 
        https://ratsgo.github.io/deep%20learning/2017/10/02/softmax/
        """
        num_class = grad_output.shape[1]

        jacobian_matrix = np.zeros((num_class, num_class))
        # jacobian : 모든 행렬 원소들이 1차 미분 계수로 구성된 행렬
        for i in range(num_class):
            for j in range(num_class):
                if i == j:
                    jacobian_matrix[i, j] = self.probs[i, j] * (1.0 - self.probs[i, j])
                else:
                    jacobian_matrix[i, j] = -self.probs[i, j] * self.probs[i, j]
        grad_output = np.dot(grad_output, jacobian_matrix)
        return grad_output


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

            # MSE
            loss += np.sum((input - y[i]) ** 2)

            # backpropagation
            grad_output = 2 * (input - y[i])  # derivative of MSE
            for layer in reversed(self.layers):
                grad_output = layer.backward(grad_output, learning_rate)
            print("Epoch %d loss: %.4f" % (epoch+1, loss / X.shape[0]))

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
    #FIXME: ConvLayer 여러개 쌓아서 학습 잘 되는지 확인
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
