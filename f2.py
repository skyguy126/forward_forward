import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Define the device to use (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gen_mnist_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        # MNIST (mean pixel value, std dev pixel value) for normalization.
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
    ])

    train_loader = DataLoader(
        MNIST('./data/', train=True, download=True, transform=transform),
        batch_size=train_batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        MNIST('./data/', train=False, download=True, transform=transform),
        batch_size=test_batch_size,
        shuffle=False
    )

    return train_loader, test_loader

def overlay_y_on_x(x, y):

    # this implementation replaces the first 10 pixels
    # of the mnist image with the label itself, which
    # is encoded as a one-hot vector. This is necessary
    # because compared to traditional backpropagation, we
    # cannot compute the loss at the end of the network
    # and propagate it backwards. Each layer is treated 
    # as a mutually exclusive entity therefore it must
    # have all the necessary information to train its
    # neurons.

    # TODO: is there a better way to do this rather than
    # mangle the first 10 pixels of the image??

    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

class Network(torch.nn.Module):
    
    def __init__(self, dims):
        
        super().__init__()
        
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d+1])]

        # Linear classifier (dense output with 10 nodes)
        self.classifier = torch.nn.Linear(dims[-1], 10)
        self.classifier.to(device) # cuda

    def predict(self, x):

        x = x.to(device) # cuda

        # run goodness for each layer in net
        h = overlay_y_on_x(x, label)
        for layer in self.layers:
            h = layer(h)

        # feed into linear classifier
        z = self.classifier(h)
        return torch.argmax(z, dim=1)

    def train(self, x_pos, x_neg, y):

        # cuda
        x_pos = x_pos.to(device)
        x_neg = x_neg.to(device)
        y = y.to(device)

        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)

        # train the linear classifier (on positive samples only)
        self.train_linear_classifier(h_pos.detach(), y)

    def train_linear_classifier(self, h, labels):
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.01)
        num_epochs = 1000 # tune this hyperparameter

        for epoch in range(num_epochs):
            
            optimizer.zero_grad()
            z = self.classifier(h)
            loss = criterion(z, labels)
            loss.backward()
            optimizer.step()

# torch.nn.Linear is used to implement a dense
# (fully connected) layer. It applies a linear
# transformation to the input data defined as
# y = xA^T + b [weight and bias].
class Layer(torch.nn.Linear):

    def __init__(self, in_features, out_features):

        # the bias term allows the neuron to learn to activate even
        # when the weighted input is very small. Prevents "dead" neurons.
        super().__init__(in_features, out_features, bias=True, device=device, dtype=None) # cuda

        self.relu = torch.nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.03)
        self.num_epochs = 1000

        # learning rate is a hyperparameter.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.03)

        # this is the threshold for the loss function. If the 
        # goodness is above the threshold or lower than the 
        # threshold we'll update the weights accordingly.
        # this is a tunable hyperparameter.
        self.threshold = 2.0
    
    def forward(self, x):
        # normalize the input tensor and add epsilon to prevent
        # divide by zero errors. p specifies Euclidean norm, and
        # dim specifies to normalize row-wise. 
        x_norm = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-4)

        # run the forward pass with a ReLU function. 
        # TODO: maybe we can use the smooth-ReLU here for
        # consistency
        return self.relu(
            torch.mm(x_norm, self.weight.T) + self.bias.unsqueeze(0)
        )

    def train(self, x_pos, x_neg):
        
        for i in range(self.num_epochs):
            # calculate the goodness value for both the positive
            # and negative samples. the goodness function is defined
            # to be the sum of squares of the ReLU neurons in this
            # layer. ff normalizes the value to prevent the next
            # hidden layer from trivally using the length of vector 
            # to distinguish between positive and negative data.

            # two forward passes, one for positive and one for negative data.
            # this layer's objective function's objective is to maintain high goodness
            # for positive data and low goodness for negative data. The paper
            # describes goodness as sum of the squared activities in a layer.

            # x_pos is a vector containing a number of elements equal to batch size.

            goodness_value_pos = self.forward(x_pos).pow(2).mean(1) # positive samples
            goodness_value_neg = self.forward(x_neg).pow(2).mean(1) # negative (corrupted) samples

            # the loss function is designed to push the goodness score of positive
            # data above the threshold hyperparameter, and goodness of negative data
            # below the threshold.

            # concatenate the positive and negative goodness vectors and apply
            # the threshold. this outputs a vector of length (2 * batch size).
            # this creates a single vector that represents the errors of the
            # network for both positive and negative samples.
            pos_neg_goodness_error_vector = torch.cat([
                -goodness_value_pos + self.threshold,
                goodness_value_neg - self.threshold
            ])

            # perform a smooth approximation of ReLU so we can compute derivatives.
            # ReLU is non-differentiable at x=0 and log(1 + e^x) is a smooth approx
            # put the function on Desmos to prove it to yourself. smooth_relu is
            # the error vector fed through the ReLU function so it is still a vector.
            # we don't technically need the error vector itself as we only care about
            # the gradient since that is what updates the weights and biases. The error
            # vector values are useful for monioring progress and such.
            loss = torch.log(1 + torch.exp(pos_neg_goodness_error_vector)).mean()

            # zero out the gradients from the previous batch, so we can start
            # fresh for the current batch. pytorch by default accumulates
            # gradients.
            self.optimizer.zero_grad()

            # pytorch keeps track of all the log, exp, and other functions applied
            # to the vector. therefore when you call backward() it is able to go
            # through the graph and compute the gradient of the vector. this function
            # call stores the gradient in the .grad portion of the weights and biases.
            loss.backward()

            # the optimizer uses the .grad portion of the weights and biases to update
            self.optimizer.step()

        # need to call detach to remove them from the computational graph, we don't 
        # care about the gradients and such after returning from the train function
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

if __name__ == "__main__":

    torch.manual_seed(1234)
    train_loader, test_loader = gen_mnist_loaders()

    network = Network([784, 500, 500])

    # load the positive examples
    x, y = next(iter(train_loader))
    x_pos = overlay_y_on_x(x, y)

    # load the negative examples
    # TODO: no idea how these two lines work
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])

    network.train(x_pos, x_neg, y)
    print('train error:', 1.0 - network.predict(x).eq(y).float().mean().item())

    # load test dataset
    x_te, y_te = next(iter(test_loader))
    print('test error:', 1.0 - network.predict(x_te).eq(y_te).float().mean().item())
