import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        show_progress: bool = False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # creating dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    train_loss_lst = []
    eval_loss_lst = []

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        network.train()
        epoch_train_loss = 0.0
        train_batches = 0

        if show_progress:
            train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        else:
            train_loader_iter = train_loader

        # training
        for batch in train_loader_iter:
            inputs, targets, name, path = batch
            inputs, targets = inputs.to(device).float(), targets.to(device).long()
            #targets = targets.unsqueeze(1)
            optimizer.zero_grad()

            outputs = network(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_batches += 1

        average_train_loss = epoch_train_loss / train_batches
        train_loss_lst.append(average_train_loss)


        network.eval()
        epoch_eval_loss = 0.0
        eval_batches = 0

        # evaluating
        with torch.no_grad():
            for batch in eval_loader:
                inputs, targets, name, path = batch
                inputs, targets = inputs.to(device).float(), targets.to(device).long()
                #targets = targets.unsqueeze(1)
                outputs = network(inputs)
                loss = loss_fn(outputs, targets)

                epoch_eval_loss += loss.item()
                eval_batches += 1

        average_eval_loss = epoch_eval_loss / eval_batches
        eval_loss_lst.append(average_eval_loss)

    torch.save(network.state_dict(), "model.pth")

    return train_loss_lst, eval_loss_lst