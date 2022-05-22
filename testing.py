import torch
import torch.nn.functional as F

# WandB – Import the wandb library
import wandb

def test(model, device, test_loader, classes):
    # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc 
    # which behave differently in training and evaluation mode
    model.eval()
    correct = 0

    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            # Load the input features and labels from the test dataset
            data, target = data.to(device), target.to(device)

            # Make predictions: Pass image data from test dataset, 
            # make predictions about class image belongs to (0-9 in this case)
            output = model(data)

            # Get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            # WandB – Log images in your test dataset automatically, 
            # along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
            example_images.append(
                wandb.Image(
                    data[0], caption="Pred: {} Truth: {}".format(classes[pred[0].item()], classes[target[0]])
                )
            )

    accuracy = 100. * correct / len(test_loader.dataset)
    # WandB – wandb.log(a_dict) logs the keys and values of the dictionary passed in and associates the values with a step.
    # You can log anything by passing it to wandb.log, including histograms, custom matplotlib objects, images, video, text, tables, html, pointclouds and other 3D objects.
    # Here we use it to log test accuracy, loss and some test images (along with their true and predicted labels).
    wandb.log({
        "Examples": example_images,
        "Test Accuracy": accuracy
    })

    return accuracy