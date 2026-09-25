import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # first layer: input layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # second layer: hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # third layer: output layer
            torch.nn.Linear(20, num_outputs),
            # torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx] # return a tuple of features and labels

def compute_accuracy(model, dataloader):
    model.eval()
    correct = 0.0
    total_examples = 0
    for idx, (features, labels) in enumerate(dataloader):
        
        with torch.no_grad():
            logits = model(features)
        
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

if __name__ == "__main__":
    torch.manual_seed(123)
    model = NeuralNetwork(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    num_epochs = 3


    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])

    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])

    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
   
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):

            logits = model(features)
            
            loss = F.cross_entropy(logits, labels) # Loss function
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            ### LOGGING
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                f" | Batch {batch_idx+1:03d}/{len(train_loader):03d}"
                f" | Train/Val Loss: {loss:.2f}")

    model.eval()
    # Optional model evaluation
    with torch.no_grad():
        outputs = model(X_train)
    print(outputs)

    torch.set_printoptions(sci_mode=False)
    probas = torch.softmax(outputs, dim=1)
    print(probas)

    predictions = torch.argmax(probas, dim=1)
    print(predictions)

    print(predictions == y_train)
    print(torch.sum(predictions == y_train))

    print(compute_accuracy(model, train_loader))
    print(compute_accuracy(model, test_loader))

    torch.save(model.state_dict(), "model.pth")