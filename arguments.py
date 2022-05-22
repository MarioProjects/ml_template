import argparse

class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(
    description='Neurovías Training', formatter_class=SmartFormatter
)

parser.add_argument(
    '--run_name', type=str, default="baseline",
    help='Name of the run'
)

parser.add_argument(
    '--batch_size', type=int, default=128,
    help='Input batch size for data (default: 128)'
)

parser.add_argument(
    '--epochs', type=int, default=200,
    help='Number of epochs to train (default: 200)'
)

parser.add_argument(
    '--learning_rate', type=float, default=0.1,
    help='Learning rate (default: 0.1)'
)

parser.add_argument(
    '--momentum', type=float, default=0.9,
    help='SGD momentum (default: 0.9)'
)

parser.add_argument(
    "--seed", type=int, default=42,
    help='Random seed (default: 42)'
)

args = parser.parse_args()

# Log the hyperparameters
print("\nHyperparameters:")
for argument in args.__dict__:
    print("- {}: {}".format(argument, args.__dict__[argument]))
print("\n")