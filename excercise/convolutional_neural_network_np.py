import numpy as np
import torchvision

'''  Memo
np.matmul(self.filters[f, ...], receptive_field) == self.filters[f, ...] @ receptive_field

matmul : matrix multiplication
dot : inner product
* : Hadamard product

np.array([[1, 3], [2, 4]])  # row vector
== [[1, 3]
    [2, 4]]

    
# https://jimmy-ai.tistory.com/104
# np.dot(a, b) : element-wise multiplication and sum
# !!! but 2D * 2D : normal matrix multiplicaiton. not element-wise

왼쪽 array의 last axis, 오른쪽 array의 second last axis 끼리 곱한다고 되어있지만,
쉽게 설명하면 왼쪽 array의 각 행, 오른쪽 array의 각 열끼리
순서대로 내적을 수행한 결과를 반환

a = np.array([[1, 3], [2, 4]])
b = np.array([[[1, 0], [0, 1]], [[0, 0], [0, 0]]])
np.dot(a, b)
(2 x 2) x (2, 2, 2) => (2, 2, 2)

c = np.array([[[1, 0, 0], [0, 1, 0]], [[-1, 0, 0], [0, -1, 0]], [[0, 0, 0], [0, 0, 0]]])
np.dot(a, c)
(2 x 2) x (3, 2, 3) => (2, 3, 3)


# np.matmul과 np.dot이 같은 결과를 나타내는 경우
1. 1D * 1D (inner product)
2. 2D * 2D (2D matrix multiplication)
3. 2D * 1D  or  1D * 2D (matrix * vector)


<<< Model Structure >>>
ConvLayer
ReLU
MaxPoolLayer
FlattenLayer
DenseLayer
Softmax

'''


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

        batch_size, height, width, _ = padded_inputs.shape
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
                        self.outputs[b, i, j, f] = (self.filters[f, ...] * receptive_field.squeeze()).sum()
                        
                        if (self.filters[f, ...] * receptive_field.squeeze()).sum() > 0.1:
                            print(f'conv output : {(self.filters[f, ...] * receptive_field.squeeze()).sum()}')
                            #FIXME: pytorch는 conv2d output max가 커져봤자 4~5이지 100이상 되지 않았음.
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

        for b in range(batch_size):  # b
            for i in range(self.outputs.shape[1]):  # h
                for j in range(self.outputs.shape[2]):  # w
                    for f in range(self.num_filters):  # c
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
                                          ] += np.expand_dims(self.filters[f, ...] * grad_input[b, i, j, f], axis=2)
        
        self.filters -= learning_rate * grad_filters  # weight update.
        grad_output = padded_grad_inputs[
                                         :,
                                         self.padding : -self.padding if self.padding != 0 else None, # remove padding
                                         self.padding : -self.padding if self.padding != 0 else None, # remove padding
                                         :
                                        ]
        
        return grad_output


class ReLU:
    def __init__(self):
        ...

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
    def __init__(self):
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
        self.weights -= learning_rate * np.dot(self.input.T, grad_output)
        self.bias -= learning_rate * np.sum(grad_output, axis=0, keepdims=True)
        
        return np.dot(grad_output, self.weights.T)


class SoftmaxLayer:
    def __init__(self):
        self.probs = None

    def forward(self, inputs):
        exp_inputs = np.exp(inputs)
        self.probs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        #FIXME: overflow !!!
        return self.probs

    def backward(self, grad_output, _):
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
                    jacobian_matrix[i, j] = self.probs[0, j] * (1.0 - self.probs[0, j])
                else:
                    jacobian_matrix[i, j] = -self.probs[0, j] ** 2
        grad_output = np.dot(grad_output, jacobian_matrix)
        return grad_output


class CNN:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)

    def train(self, X, y, num_epochs, learning_rate, batch_size):
        for epoch in range(num_epochs):
            loss = 0

            for i in range(len(X) // batch_size):
                input = np.expand_dims(X[i], axis=0)
                
                target = y[i]

                for layer in self.layers:
                    if i == 3 or input.max() > 1000:
                        1
                    input = layer.forward(input)  # input 이지만 output임 (...)

                # MSE
                loss += np.sum((input - target) ** 2) / input.shape[0]

                # backpropagation
                grad_output = 2 * (input - target)  # derivative of MSE

                for layer in reversed(self.layers):
                    grad_output = layer.backward(grad_output, learning_rate)

            print("Epoch %d loss: %.4f" % (epoch + 1, loss / X.shape[0]))

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            input = np.expand_dims(X[i], axis=0)
            for layer in self.layers:
                input = layer.forward(input)
            prediction = np.argmax(input)
            predictions.append(prediction)
        return np.array(predictions)



if __name__ == "__main__":

    bs = 1  # batch size
    n_iter = 2000
    small_dataset_size = bs * n_iter  
    # bs1로 200개 샘플 학습 : loss는 줄어드나 test acc는 10%
    # bs1로 2000개 샘플 학습 : loss는 줄어드나 test acc는 17%
    # bs1로 20000개 샘플 학습 : loss는 줄어들다 조금 오르다 test acc는 93%

    epochs = 10
    lr = 0.01

    mnist_train = torchvision.datasets.MNIST(root='MNIST_data/',
                                             train=True,
                                             download=True)

    mnist_test = torchvision.datasets.MNIST(root='MNIST_data/',
                                            train=False,
                                            download=True)
    
    X_train = np.expand_dims(mnist_train.train_data.numpy(), axis=3)[:small_dataset_size, ...]
    y_train_tmp = mnist_train.train_labels.numpy()[:small_dataset_size, ...]

    X_test = np.expand_dims(mnist_test.test_data.numpy(), axis=3)[:small_dataset_size, ...]
    y_test_tmp = mnist_test.test_labels.numpy()[:small_dataset_size, ...]

    X_train = X_train / 255
    X_test = X_test / 255

    y_train = np.zeros((small_dataset_size, 10))
    y_test = np.zeros((small_dataset_size, 10))
    for i in range(small_dataset_size):
        y_train[i, y_train_tmp[i]] = 1
        y_test[i, y_test_tmp[i]] = 1

    cnn = CNN()    
    #TODO: ConvLayer 여러개 쌓아서 학습 잘 되는지 확인, pytorch code 성능과 비교
    cnn.add(ConvLayer(8, 3, 1, 0))  # num_filters, filter_size, stride, padding
    cnn.add(ReLU())
    cnn.add(MaxPoolLayer(2, 2))
    cnn.add(FlattenLayer())
    cnn.add(DenseLayer(10))
    cnn.add(SoftmaxLayer())

    cnn.train(X_train, y_train, num_epochs=epochs, learning_rate=lr, batch_size=bs)

    predictions = cnn.predict(X_test)
    print(f"test set accuracy {100 * sum(y_test_tmp == predictions) / len(predictions):.2f}%")