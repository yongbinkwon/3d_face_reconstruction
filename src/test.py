import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


#En annen måte å lage nettverk på utenom sequential. Mer fleksibel men litt mer kode. Du kan bruke sequential hvis du heller vil det.
#Du kan se at jeg bruker noe som heter "class", og kanskje litt ukjente funksjonnavn som aldri blir kalt eksplisitt (__init__ og __call__).
#Dette er en annen måte å programmere på som heter objekt-orientert programmering hvis du ikke har vært innom det (java skrives på denne måten f.eks)
#Du kan spørre om det hvis du vil ha en gjennomgang av det ellers kan du bare bruke det/bruke sequential for nettverk.
class Model(nn.Module):
    def __init__(self, n_in, n_out):
        super(Model, self).__init__()
        #Her er layers
        self.fc1 = nn.Linear(n_in, 5)
        self.fc2 = nn.Linear(5, n_out)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__=='__main__':
    #Sjekker om du har CUDA (krever nvidia grafikkort) til å bruke, hvis ikke bruker den cpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    #Hvis cuda er tilgjengelig finner programmet den mest optimale konvolusjons-algoritmen. Er ikke relevant fordi du ikke bruker convolutional layers tror jeg.
    torch.backends.cudnn.benchmark = True

    #Model blir initialisert og da blir __init__ funksjonen kalt
    model = Model(3, 3).to(device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    
    num_epochs = 10

    #X er input Y er output
    X = [[1, 2, 3], [5, 1, 3], [4, 1, 8]]
    Y = [[2, 3, 1], [12, 4, 2], [8, 3, 5]]

    #Parametere å bruke for dataloaderen. 
    #Batch size har du kanskje lært om alt.
    #Shuffle: Hvis denne er True så blir dataene stokket før de blir hentet. Dette er ikke så relevant for batch size=1 men for større batch sizes betyr det at batchene er annerledes for hver epoke.
    #num workers: Sier hvor mange forskjellige "workers"/tråder(?) som jobber med å hente data fra datasettet. Hvis denne er 0 så er det bare hoved-tråden som henter data. Hvis num_workers = 2 vil det si at to tråder jobber i parallell med å hente data
    params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 0
    }

    #Her blir X og Y gjort om til et dataset ("TensorDataset") og dataen fra dette datasettet blir hentet ut av en dataloader ("DataLoader")
    dataloader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)), **params)

    for epoch in range(num_epochs):
        batch_loss = 0.0
        for input, ground_truth in dataloader:
            optimizer.zero_grad()

            output = model(input)
            loss = loss_function(output, ground_truth)
            loss.backward()
            optimizer.step()

            print(f"epoch: {epoch+1} - loss: {loss.item()}")
            
    print("done :)")