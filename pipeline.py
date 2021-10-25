import torch
from torchvision import datasets, transforms
from torch import nn, optim
import matplotlib.pyplot as plt


class Pipeline:
    def __init__(self, network, learning_rate):
        """
        Load Data, initialize a given network structure and set learning rate
        """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        # Download and load the training data
        trainset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

        # Download and load the test data
        testset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

        self.model = network.to(self.device)

        """
        Set appropriate loss function such that learning is equivalent to minimizing the
        cross entropy loss. Note that we are outputting log-softmax values from our networks,
        not raw softmax values, so just using torch.nn.CrossEntropyLoss is incorrect.
        
        All networks output log-softmax values (i.e. log probabilities or.. likelihoods.). 
        """
        self.lossfn = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.num_train_samples = len(self.trainloader)
        self.num_test_samples = len(self.testloader)

    def view_batch(self):
        """
        Display first batch of images from trainloader in 8x8 grid

        Return:
           1) A float32 numpy array (of dim [28*8, 28*8]), containing a tiling of the batch images,
           place the first 8 images on the first row, the second 8 on the second row, and so on

           2) An int 8x8 numpy array of labels corresponding to this tiling
        """
        [images, labels] = [x for x in self.trainloader][0]
        image_grid = images.view(8, 8, 28, 28)
        image_grid = image_grid.permute(0, 2, 1, 3)
        labels_grid = labels.view(8, 8)
        return image_grid.reshape(8 * 28, 8 * 28), labels_grid

    def train_step(self):
        self.model.train()
        for images, labels in self.trainloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            log_ps = self.model(images)
            loss = self.lossfn(log_ps, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return

    def train_epoch(self):
        self.model.train()
        for images, labels in self.trainloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            log_ps = self.model(images)
            loss = self.lossfn(log_ps, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return

    def eval(self):
        self.model.eval()
        accuracy = 0
        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                log_ps = self.model(images)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        return accuracy / self.num_test_samples



