import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def feedforward(self, X):
        # Calculate hidden layer output
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Calculate output layer output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backpropagation(self, X, y, learning_rate):
        # Calculate output layer error and delta
        d2 = (self.a2 - y) * self.sigmoid_derivative(self.z2)
        
        # Calculate hidden layer error and delta
        d1 = np.dot(d2, self.W2.T) * self.sigmoid_derivative(self.z1)
        
        # Update weights and biases
        self.W2 -= learning_rate * np.dot(self.a1.T, d2)
        self.b2 -= learning_rate * np.sum(d2, axis=0, keepdims=True)
        self.W1 -= learning_rate * np.dot(X.T, d1)
        self.b1 -= learning_rate * np.sum(d1, axis=0)
        
    def train(self, X, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            # Perform forward and backward propagation for each training example
            for i in range(X.shape[0]):
                x_i = X[i]
                y_i = y[i]
                self.feedforward(x_i)
                self.backpropagation(x_i, y_i, learning_rate)
                
            # Print training loss every 100 epochs
            if epoch % 100 == 0:
                loss = np.mean((self.feedforward(X) - y) ** 2)
                print(f"Epoch {epoch}, loss {loss:.4f}")
        print(f"Epoch {epoch}, loss {loss:.4f}")

# Generate some random data for training
X = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y = np.array([0, 1, 1, 0])

# Create and train the MLP
mlp = MLP(input_size=2, hidden_size=10, output_size=1)
mlp.train(X, y, num_epochs=2000, learning_rate=0.001)
