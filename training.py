def train(model, device, train_loader, optimizer, criterion):
    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.train()
    correct = 0
    
    # We loop over the data iterator, and feed the inputs to the network and adjust the weights.
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # Load the input features and labels from the training dataset
        data, target = data.to(device), target.to(device)
        
        # Reset the gradients to 0 for all learnable weight parameters
        optimizer.zero_grad()
        
        # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-9 in this case)
        output = model(data)
        
        # Define our loss function, and compute the loss
        loss = criterion(output, target)
        
        # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
        loss.backward()
        
        # Update the neural network weights
        optimizer.step()

        # Get metrics
        # Get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    accuracy = 100. * correct / len(train_loader.dataset)
    return accuracy


def get_current_lr(optimizer):
    """
    Gives the current learning rate of optimizer
    :param optimizer: Optimizer instance
    :return: Learning rate of specified optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']