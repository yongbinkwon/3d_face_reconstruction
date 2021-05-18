import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np


# En annen måte å lage nettverk på utenom sequential. Mer fleksibel men litt mer kode. Du kan bruke sequential hvis du heller vil det.
# Du kan se at jeg bruker noe som heter "class", og kanskje litt ukjente funksjonnavn som aldri blir kalt eksplisitt (__init__ og __call__).
# Dette er en annen måte å programmere på som heter objekt-orientert programmering hvis du ikke har vært innom det (java skrives på denne måten f.eks)
# Du kan spørre om det hvis du vil ha en gjennomgang av det ellers kan du bare bruke det/bruke sequential for nettverk.
def read_file(file):
    # åpne fil og dele opp i elementer
    f = open(file, "r")
    x = []
    for line in f.readlines():
        for elem in line.split(","):
            x.append(float(elem))
    return x

def normalize_input(input):

    def normalize(entry):
        return [entry[0]/120.0, entry[1]/120.0, entry[2]/360.0]

    return list(map(normalize, input))


def chunks(lst, n):
    # splitte elementer opp i repspektive variablgrupper
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class Model(nn.Module):
    def __init__(self, n_in, n_out):
        super(Model, self).__init__()
        # Her er layers
        self.fc_in = nn.Linear(n_in, 16)
        self.bn_in = nn.BatchNorm1d(num_features=16)
        self.fc_hidden1 = nn.Linear(16, 256)
        self.bn_hidden1 = nn.BatchNorm1d(num_features=256)
        self.fc_hidden2 = nn.Linear(256, 1024)
        self.bn_hidden2 = nn.BatchNorm1d(num_features=1024)
        self.fc_repeat = nn.Linear(1024, 1024)
        self.bn_repeat = nn.BatchNorm1d(num_features=1024)
        self.fc_out = nn.Linear(1024, n_out)

    def forward(self, x):
        x = F.relu(self.bn_in(self.fc_in(x)))
        x = F.relu(self.bn_hidden1(self.fc_hidden1(x)))
        x = F.relu(self.bn_hidden2(self.fc_hidden2(x)))
        x = F.relu(self.bn_repeat(self.fc_repeat(x)))
        #x = F.relu(self.fc_repeat(x))
        #x = F.relu(self.fc_repeat(x))
        #x = F.relu(self.fc_repeat(x))
        #x = F.relu(self.fc_repeat(x))
        x = self.fc_out(x)
        return x

class DaylightDataset(Dataset):
    #train_dim HxW (12x4)
    def __init__(self, train_file, train_dim, device, gt_file="", gt_length=""):
        self.train_data = np.array([[[]]])
        self.gt = np.array([[]])
        train_data_len = train_dim[0]*train_dim[1]
        with open(train_file, "r") as f:
            current_house_dim = (1, train_dim[0], train_dim[1])
            current_house = np.empty(current_house_dim)
            for i, line in enumerate(f):
                hei = [float(entry) for entry in line.split(", ")]
                current_house[0][i//train_dim[1]][i%train_dim[1]] = np.array(hei, dtype=np.float)
                if i%train_data_len == 0:
                    self.train_data = np.append(self.train_data, current_house, axis=0)
                    current_house = np.empty(train_dim)
        self.train_data = torch.tensor(self.train_data, dtype=torch.float32, device=device)


if __name__ == '__main__':
    # Sjekker om du har CUDA (krever nvidia grafikkort) til å bruke, hvis ikke bruker den cpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Hvis cuda er tilgjengelig finner programmet den mest optimale konvolusjons-algoritmen. Er ikke relevant fordi du ikke bruker convolutional layers tror jeg.
    torch.backends.cudnn.benchmark = True

    batch_size = 2

    # Model blir initialisert og da blir __init__ funksjonen kalt
    model = Model(3, 4426).to(device)

    loss_function = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    """
    x_test = torch.tensor((16.043, 16.0055, 179.7168), dtype=torch.float32)
    print(model(x_test))
    """

    num_epochs = 1000

    variabler = list(chunks(read_file("old_data/variabler.txt"), 3))
    resultater = list(chunks(read_file("old_data/resultater.txt"), 4426))
    print(len(resultater))
    print(len(variabler))

    # X er input Y er output
    X = normalize_input(variabler)
    Y = resultater

    test = DaylightDataset("./old_data/variabler.txt", (12,4), device)
    print(test.train_data)

    # Parametere å bruke for dataloaderen.
    # Batch size har du kanskje lært om alt.
    # Shuffle: Hvis denne er True så blir dataene stokket før de blir hentet. Dette er ikke så relevant for batch size=1 men for større batch sizes betyr det at batchene er annerledes for hver epoke.
    # num workers: Sier hvor mange forskjellige "workers"/tråder(?) som jobber med å hente data fra datasettet. Hvis denne er 0 så er det bare hoved-tråden som henter data. Hvis num_workers = 2 vil det si at to tråder jobber i parallell med å hente data
    params = {
        'batch_size': batch_size,  # vanlig med 16, 32, 64 x**2
        'shuffle': True,
        'num_workers': 0
    }

    # Her blir X og Y gjort om til et dataset ("TensorDataset") og dataen fra dette datasettet blir hentet ut av en dataloader ("DataLoader")
    dataloader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32, device=device),
                                          torch.tensor(Y, dtype=torch.float32, device=device)), **params)

    batches_in_epoch = len(X)/batch_size
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        count = 0.0
        for input, ground_truth in dataloader:
            optimizer.zero_grad()

            output = model(input)
            loss = loss_function(output, ground_truth)
            loss.backward()
            optimizer.step()
            log_loss = []
            log_epoch = []
            epoch_loss += loss.item()*output.shape[0]
            correct = (output == ground_truth).float().sum()
        """
        if((epoch+1)%100==0):
            print(count)
            print(f"epoch {epoch+1} loss: {epoch_loss/len(X)}")
            with open("./output.txt", "w") as f:
                for entry1, entry2 in zip(output[0], output[1]):
                    f.write(str((entry1.item()+entry2.item())/2.0))
                    f.write("\n")
            with open("./gt.txt", "w") as f:
                for entry1, entry2 in zip(ground_truth[0], ground_truth[1]):
                    f.write(str((entry1.item()+entry2.item())/2.0))
                    f.write("\n")
            with open("./loss.txt", "w") as f:
                f.write(str(loss.item()))
        """
        

    #torch.save(model.state_dict(), "C:/Users/tobia/Documents/Test/48ptmodell.pth")
    print("done :)")

