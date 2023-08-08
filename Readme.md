# excercise with basic deep learning models and pytorch templates

├─0_dataloader  
├─1_template  
│  ├─config  
│  ├─model  
│  ├─train  
│  └─utils  
├─excercise  
└─MNIST_data  
    └─MNIST  

## mlp with numpy

## mlp with pytorch


## Tensorflow - Keras
딥러닝 모델 연산 최적화
@tf.fuction
@tf.fuction(git_compile=True)


## general modeling convention? tips
attention value의 variance가 높으므로 sqrt로 나누어준다.

residual block의 ensemble유사한 효과를 depth에 따라 크게 주는.  
즉 drop rate를 점차 커지게 한다.
https://paperswithcode.com/method/stochastic-depth

## when deep learning model returns NaN
possible reasons
1. high learning rate
2. division by zero

possible solutions
1. lower learning rate
2. gradient clipping
3. normalize data
4. check input and target has invalid values
5. check nan using ```torch.autograd.detect_anomaly(True)``` at the beginning of my script to get a stack trace
6. print gradient in a layer using ```model.fc1.weight.grad```
7. add a small epsilon (1e-6) to get rid of the 'nan' loss value
