import numpy
import pickle
import numpy as np
import time
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset
from torchvision.transforms import transforms

Batch1 = "/Users/ansonwong/data/cifar-10-batches-py/data_batch_1"
Batch2 = "/Users/ansonwong/data/cifar-10-batches-py/data_batch_2"
Batch3 = "/Users/ansonwong/data/cifar-10-batches-py/data_batch_3"
Batch4 = "/Users/ansonwong/data/cifar-10-batches-py/data_batch_4"
Batch5 = "/Users/ansonwong/data/cifar-10-batches-py/data_batch_2"
TestBatch = "/Users/ansonwong/data/cifar-10-batches-py/test_batch"

Batches = [Batch1, Batch2, Batch3, Batch4, Batch5, TestBatch]
for count, batch in enumerate(Batches):
    with open(batch, 'rb') as fo:
        Batches[count] = pickle.load(fo, encoding='bytes')

Batch1 = Batches[0]
Batch2 = Batches[1]
Batch3 = Batches[2]
Batch4 = Batches[3]
Batch5 = Batches[4]
TestBatch = Batches[5]
lr = 0.000005
batch_size = 128
writer = SummaryWriter(filename_suffix=f"batch_size_{batch_size},lr_{lr}",
                       comment=f"batch_size_{batch_size},lr_{lr}")
device = torch.device("mps")

Trainbatch = np.concatenate((Batch1[b'data'], Batch2[b'data'], Batch3[b'data'], Batch4[b'data'], Batch5[b'data']),
                            axis=0)
Trainlabel = np.concatenate((Batch1[b'labels'], Batch2[b'labels'], Batch3[b'labels'], Batch4[b'labels'],
                             Batch5[b'labels']), axis=0)
Trainbatch = Trainbatch.reshape(50000, 3, 32, 32)
TestBatch[b'data'] = TestBatch[b'data'].reshape(10000, 3, 32, 32)

TrainData = torch.from_numpy(Trainbatch)
TrainLabel = torch.tensor(Trainlabel).to(device)
TestBatchData = torch.from_numpy(TestBatch[b'data']).to(device)
TestBatchLabel = torch.tensor(numpy.array(TestBatch[b'labels']).reshape(10000, 1))
TrainLabel = F.one_hot(TrainLabel, num_classes=10)

Traindataset = TensorDataset(TrainData, TrainLabel)
Testdataset = TensorDataset(TestBatchData, TestBatchLabel)
TrainLoader = torch.utils.data.DataLoader(Traindataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                          drop_last=True)
TestLoader = torch.utils.data.DataLoader(Testdataset, batch_size=1, shuffle=False, num_workers=0)

transform = transforms.Compose(
    [transforms.PILToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomRotation((-10, 10), expand=True, ),
     transforms.RandomHorizontalFlip(),
     transforms.RandomCrop((32, 32), padding=1),
     transforms.Resize((32, 32), antialias=True)
     ])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.network(x)
        return output


# %%
network = Net().to(device)
# network.load_state_dict(torch.load("/models/5_model.pth"))
loss_function = nn.MSELoss()
trainresult = {
    "Airplane": [0, 0],
    "Automobile": [0, 0],
    "Bird": [0, 0],
    "Cat": [0, 0],
    "Deer": [0, 0],
    "Dog": [0, 0],
    "Frog": [0, 0],
    "Horse": [0, 0],
    "Ship": [0, 0],
    "Truck": [0, 0],
}
accuracy = []
optimizer = torch.optim.Adam(network.parameters(), lr=lr)
network.load_state_dict(torch.load("blankmodel/blank.pth")[0])
optimizer.load_state_dict(torch.load("blankmodel/blank.pth")[1])
starttime = time.time()
testtime = 0
_iter = 0

for epoch in range(0, 100):
    traincorrect = 0
    network.train()
    train_loss = 0
    for u, (_inputs, labels) in enumerate(TrainLoader):
        _inputs = _inputs.float()
        inputs = []
        for count, k in enumerate(_inputs):
            k.numpy()
            k = np.transpose(k, axes=[1, 2, 0])
            k = np.uint8(k)
            k = Image.fromarray(k, mode="RGB")
            k = transform(k)
            k = k.numpy()
            inputs.append(k)
        inputs = np.array(inputs)
        inputs = torch.from_numpy(inputs).to(device)
        inputs = inputs.float()
        labels = labels.float()
        outputs = network(inputs)
        loss = loss_function(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        for c, i in enumerate(outputs):
            sth = torch.argmax(labels[c])
            thisthingy = list(trainresult.keys())[sth]
            if torch.argmax(i) == sth:
                traincorrect += 1
                trainresult[thisthingy][0] += 1
            trainresult[thisthingy][1] += 1
        print(
            f"\rTraining epoch {epoch + 1}, {round(u / len(TrainLoader) * 100, 5)}% --- Accuracy : {round(traincorrect / ((u + 1) * batch_size), 5)} --- Loss : {round(train_loss / (u + 1), 5)} ",
            end="")
        writer.add_scalar("Train Loss / Data", loss.item(), _iter * batch_size)
        _iter += 1
    accuracy.append(traincorrect / len(TrainData))
    train_loss = train_loss / len(TrainLoader)
    print(f"\rEpoch {epoch + 1} -- Train loss {train_loss} -- Accuracy {traincorrect / len(TrainData)}")
    writer.add_scalar("Train Loss / Epoch", train_loss, epoch + 1)
    writer.add_scalar("Train Accuracy / Epoch", traincorrect / len(TrainData), epoch + 1)
    writer.add_scalar("Epoch / Time", epoch + 1, int(time.time() - starttime) - testtime)
    for i in range(10):
        print(
            f"{list(trainresult.keys())[i]} --> {trainresult[list(trainresult.keys())[i]][0] / trainresult[list(trainresult.keys())[i]][1]}")
    print("\n")
    if (epoch + 1) % 5 == 0:
        testst = time.time()
        torch.save(network.state_dict(),
                   f"/Users/ansonwong/PycharmProjects/image_recognition/models/{epoch + 1}_model.pth")
        print("test:")
        network.eval()
        with torch.no_grad():
            correct = 0
            result = {
                "Airplane": [0, 0],
                "Automobile": [0, 0],
                "Bird": [0, 0],
                "Cat": [0, 0],
                "Deer": [0, 0],
                "Dog": [0, 0],
                "Frog": [0, 0],
                "Horse": [0, 0],
                "Ship": [0, 0],
                "Truck": [0, 0],
            }
            for a, (testinputs, testlabels) in enumerate(TestLoader):
                testinputs = testinputs.float()
                testoutputs = network(testinputs)
                lab = testlabels.item()
                if torch.argmax(testoutputs).item() == lab:
                    result[list(result.keys())[lab]][0] += 1
                    correct += 1
                result[list(result.keys())[lab]][1] += 1
            print(f"overall --> {correct / len(TestLoader)}")
            writer.add_scalar("Test Accuracy / Epoch", correct / len(TestLoader), epoch + 1)
            for i in range(10):
                print(
                    f"{list(result.keys())[i]} --> {result[list(result.keys())[i]][0] / result[list(result.keys())[i]][1]}")
        print("")
        testtime += time.time() - testst
