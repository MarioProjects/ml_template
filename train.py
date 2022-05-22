from arguments import args
from training import train, get_current_lr
from testing import test
import wandb
from models import *
import torch
import torch.optim as optim
from torchvision import datasets, transforms
# Ignore excessive warnings
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)


# WandB – Initialize a new run
wandb.init(
    project="pytorch-intro", name=args.run_name, config=args
)
# Re-run the model without restarting the runtime, unnecessary after our next release
wandb.watch_called = False


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Set random seeds and deterministic pytorch for reproducibility
# random.seed(args.seed)       # python random seed
torch.manual_seed(args.seed)  # pytorch random seed
# numpy.random.seed(args.seed) # numpy random seed
torch.backends.cudnn.deterministic = True

# Load the dataset: We're training our CNN on CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html)
# First we define the tranformations to apply to our images
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Now we load our training and test datasets and apply the transformations defined above
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    ),
    batch_size=args.batch_size, shuffle=True, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    ),
    batch_size=args.batch_size, shuffle=False, **kwargs
)

classes = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

# Initialize our model, recursively go over all modules
# and convert their parameters and buffers to CUDA tensors (if device is set to cuda)
model = ResNet18().to(device)
optimizer = optim.SGD(
    model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=5e-4
)
criterion = nn.CrossEntropyLoss()
# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# WandB – wandb.watch() automatically fetches all layer dimensions, gradients,
# model parameters and logs them automatically to your dashboard.
# Using log="all" log histograms of parameter values in addition to gradients
wandb.watch(model, log="all")

print("\n=> Starting training...\n")
for epoch in range(1, args.epochs + 1):
    train_accuracy = train(model, device, train_loader, optimizer, criterion)
    test_accuracy = test(model, device, test_loader, classes)
    print(f"Epoch: {epoch} Train Accuracy: {train_accuracy:.2f}% Test Accuracy: {test_accuracy:.2f}% LR: {get_current_lr(optimizer):.4f}")
    scheduler.step()

# WandB – Save the model checkpoint.
# This automatically saves a file to the cloud and associates it with the current run.
torch.save(model.state_dict(), "model.h5")
wandb.save('model.h5')
