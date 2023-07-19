import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--train_ratio', type=float, default=0.5, help='Ratio for training data')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')

    # Model
    parser.add_argument('--model', type=str, default='vgg16', help='Model name')
    parser.add_argument('--input_dim', type=int, default=100, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=10, help='Output dimension')

    # Training
    parser.add_argument('--learning_rate', type=float, default=0.04, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--crit', type=str, default='cross_entropy', help='Loss criterion')
    parser.add_argument('--learned_metric', type=str, default='epoch', help='Iteration Learned or Epoch Learned')
    parser.add_argument('--batch_size', type=int, default=1000, help="batch size")

    # Other configurations
    parser.add_argument('--seed', type=int, default=9203, help='Seed value')
    parser.add_argument('--result_dir', type=str, default='cs_results', help='Result directory')
    parser.add_argument('--save_result', type=bool, default=True, help='Save results')
    parser.add_argument('--n_runs', type=int, default=200, help='Number of runs')
    parser.add_argument('--ss_ratio', type=float, default=0.1, help='Ratio of subset set')

    return parser.parse_args()