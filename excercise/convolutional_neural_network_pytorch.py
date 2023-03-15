import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1, 0)
        # in_channel, out_channel, filter_size, stride, padding
        
        # self.conv2
        # self.dropout
        self.fc1 = nn.Linear(1352, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # dropout
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.softmax(x, dim=1)

        return output
        # x = F.log_softmax(x)
                
        # cnn.add(ConvLayer(8, 3, 1, 0))  # num_filters, filter_size, stride, padding
        # cnn.add(ReLU())
        # cnn.add(MaxPoolLayer(2, 2))
        # cnn.add(FlattenLayer())
        # cnn.add(DenseLayer(10))
        # cnn.add(SoftmaxLayer())
   

if __name__ == "__main__":

    # hyperparameters
    bs = 20
    epoch_num = 10
    lr = 0.01
    count_threshold = 20


    # load dataset
    train_data = torchvision.datasets.MNIST(root = './MNIST_data',
                                            train = True,
                                            download=True,
                                            transform=transforms.ToTensor())
    test_data = torchvision.datasets.MNIST(root = './MNIST_data',
                                            train = False,
                                            download=True,
                                            transform=transforms.ToTensor())


    # dataloader 
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=bs,
                                               shuffle=True
                                              )
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=bs,
                                               shuffle=False
                                              )
    first_batch_X, fisrt_batch_y = train_loader.__iter__().__next__()
    print(first_batch_X.shape, fisrt_batch_y.shape)


    # training methods setting
    is_cuda = torch.cuda.is_available()
    device =  torch.device('cuda' if is_cuda else 'cpu')
    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    

    # do training
    model.train()
    for i, epoch in enumerate(range(epoch_num)):
        count = 0
        for data, target in train_loader:
            if count > count_threshold:
                break

            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            output = model(data)

            target = F.one_hot(target, num_classes = 10).to(torch.float32)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        print(f"train epoch : {i + 1}\t loss : {loss.item():.3f}")


    # do evaluation
    model.eval()
    correct = 0
    count = 0
    for data, target in test_loader:
        if count > count_threshold:
            break
        data = data.to(device)
        target = target.to(device)
        output = model(data)

        pred = output.data.max(1)[1]

        correct += pred.eq(target.data).sum()
    print(f"test set accuracy {correct.cpu().numpy() / len(test_loader.dataset):.2f}%")
