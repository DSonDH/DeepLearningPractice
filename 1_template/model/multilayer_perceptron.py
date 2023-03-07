import numpy as np

# sample data
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

num_node_h1 = 10
num_node_h2 = 10


w1 = np.random.randn(num_node_h1, 2)  # output feature x input features
b1 = np.random.randn(1, num_node_h1)

w2 = np.random.randn(num_node_h2, num_node_h1)
b2 = np.random.randn(1, num_node_h2)

w3 = np.random.randn(1, num_node_h1)
b3 = np.random.randn(1)

epoch = 2000
learning_rate = 1
mse = []

# Neural Networks 2-2-1
for i in range(epoch):

    error = np.array([])
    result = np.array([])

    # for every sample
    for j in range(len(x)):
        hidden_1 = np.array([])
        hidden_2 = np.array([])

        # for every weight_h1
        for k in range(len(w1)):
            hidden_1 = np.append(hidden_1, 
                                 1 / (1 + np.exp(
                                              -(np.sum(x[j] * w1[k] + b1[0][k]))
                                          )
                                     )
                       )
        
        # for every weight_h2
        for m in range(len(w2)):
            hidden_2 = np.append(hidden_2, 
                                 1 / (1 + np.exp(
                                             -(np.sum(hidden_1[k] * w2[m] + b2[0][k]))
                                          )
                                     )
                       )
        
        # sigmoid output node하나니깐 for loop 없음
        output = 1 / (1 + np.exp(
                                -(np.sum(hidden_2 * w3) + b3)
                            )
                     )
        
        error = np.append(error, (y[j] - output))
        result = np.append(result, output)

        # backpropagation
        alpha3 = error[j] * output * (1 - output)
        alpha2 = alpha3 * w3 * hidden_2 * (1 - hidden_2)  # (1, 10)        
        alpha1 = alpha2 * np.matmul(w2.T, hidden_1 * (1 - hidden_1))

        w3 = w3 + learning_rate * alpha3 * hidden_2
        b3 = b3 + learning_rate * alpha3

        w2 = w2 + learning_rate * alpha2.T * hidden_1  # hidden_1 (10,)
        b2 = b2 + learning_rate * alpha2

        w1 = w1 + learning_rate * np.matmul(alpha1.T, np.expand_dims(x[j], 1).T)
        b1 = b1 + learning_rate * alpha1

    if i % 100 == 0:
        print("EPOCH : %05d MSE : %04f RESULTS : 0 0 => %04f,  0 1 => %04f,  1 0 => %04f,  1 1 => %04f"
            %(i, np.mean(error**2), result[0], result[1], result[2], result[3]))
    
    mse.append(error)

print("EPOCH : %05d MSE : %04f RESULTS : 0 0 => %04f,  0 1 => %04f,  1 0 => %04f,  1 1 => %04f"
            %(i, np.mean(error**2), result[0], result[1], result[2], result[3]))
# 3node EPOCH : 01999 MSE : 0.000781 RESULTS : 0 0 => 0.036168,  0 1 => 0.972955,  1 0 => 0.970571,  1 1 => 0.014729
# 4node EPOCH : 01999 MSE : 0.000689 RESULTS : 0 0 => 0.023938,  0 1 => 0.974317,  1 0 => 0.972405,  1 1 => 0.027582
# 10node EPOCH : 01999 MSE : 0.000454 RESULTS : 0 0 => 0.017797,  0 1 => 0.978003,  1 0 => 0.978727,  1 1 => 0.023715
# 100node EPOCH : 01999 MSE : 0.250055 RESULTS : 0 0 => 0.010722,  0 1 => 0.991978,  1 0 => 0.992148,  1 1 => 0.999989
# 10x10node EPOCH : 01999 MSE : 0.224725 RESULTS : 0 0 => 0.306537,  0 1 => 0.622432,  1 0 => 0.501447,  1 1 => 0.643290
# 10x10node EPOCH : 01999 MSE : 0.267745 RESULTS : 0 0 => 0.498835,  0 1 => 0.464600,  1 0 => 0.501165,  1 1 => 0.535400
# 10x10node EPOCH : 01999 MSE : 0.266463 RESULTS : 0 0 => 0.499264,  0 1 => 0.466158,  1 0 => 0.502119,  1 1 => 0.532649
# 10x10node EPOCH : 01999 MSE : 0.266288 RESULTS : 0 0 => 0.497849,  0 1 => 0.471138,  1 0 => 0.497660,  1 1 => 0.534095
print(mse)