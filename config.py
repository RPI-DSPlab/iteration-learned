import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--train_ratio', type=float, default=0.5, help='Ratio for training data')

    # Model
    parser.add_argument('--model', type=str, default='vgg16', help='Model name')
    parser.add_argument('--input_dim', type=int, default=100, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=10, help='Output dimension')

    # Training
    parser.add_argument('--learning_rate', type=float, default=0.04, help='Learning rate')
    parser.add_argument('--iterations', type=int, default=15000, help='Number of iterations')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--crit', type=str, default='cross_entropy', help='Loss criterion')
    parser.add_argument('--learned_metric', type=str, default='epoch', help='Iteration Learned or Epoch Learned')

    # Other configurations
    parser.add_argument('--seeds', nargs='+', type=int, default=[9203, 9304, 3456, 5210],
                        help='Seed values')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--result_dir', type=str, default='il_results', help='Result directory')
    parser.add_argument('--save_result', type=bool, default=True, help='Save results')
    parser.add_argument('--skip_training', type=bool, default=False, help='Skip calling the training function')

    return parser.parse_args()
