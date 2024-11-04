import torch
from train import training_loop
from architecture import MyCNN
from dataset import get_dataset
from torch.utils.data import DataLoader
from set_seed import set_seed
import os

def print_first_parameter(network):
    print("First Network Parameter:")
    for name, param in network.named_parameters():
        print(f"{name}:")
        print(param.data[0])
        break

if __name__ == "__main__":
    set_seed(1245)

    print(torch.cuda.is_available())
    data_dir = os.path.join(os.getcwd(), 'training_data')
    train_data, eval_data, test_data = get_dataset(data_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = MyCNN().to(device)

    #print_first_parameter(network)

    train_losses, eval_losses = training_loop(network, train_data, eval_data, num_epochs=20,
                                              batch_size=32, learning_rate=1e-4, show_progress=True)

    #print_first_parameter(network)

    for epoch, (tl, el) in enumerate(zip(train_losses, eval_losses)):
        print(f"Epoch: {epoch} --- Train loss: {tl:7.4f} --- Eval loss: {el:7.4f}")


    # test accuracy
    correct = 0
    total = 0

    test_dl = DataLoader(dataset=test_data, shuffle=False, batch_size=len(test_data))

    with torch.no_grad():
        for X, y, _, _ in test_dl:
            X, y = X.to(device), y.to(device)
            y_pred = network(X)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(correct / total)

