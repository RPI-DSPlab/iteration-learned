import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')

    parser = argparse.ArgumentParser(description='arguments to compute prediction depth for each data sample')
    parser.add_argument('--train_ratio', default=0.5, type=float, help='ratio of train split / total data split')
    parser.add_argument('--result_dir', default='./pd_results', type=str, help='directory to save ckpt and results')
    parser.add_argument('--model_dir', default='./pd_model', type=str, help='directory to save ckpt and results')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('--model', default='vgg', type=str, help='Model name')
    parser.add_argument('--get_train_pd', default=False, type=bool, help='get prediction depth for training split')
    parser.add_argument('--get_val_pd', default=True, type=bool, help='get prediction depth for validation split')
    parser.add_argument('--resume', default=False, type=bool, help='resume from the ckpt')
    parser.add_argument('--fraction', default=0.4, type=float, help='ratio of noise')
    parser.add_argument('--half', default=False, type=str,
                        help='use amp if GPU memory is 15 GB; set to False if GPU memory is 32 GB ')

    # Training
    parser.add_argument('--batch_size', type=int, default=5000, help="batch size")
    parser.add_argument('--learning_rate', type=float, default=0.04, help='Learning rate')
    parser.add_argument('--iterations', type=int, default=15000, help='Number of iterations')
    parser.add_argument('--num_epochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--crit', type=str, default='cross_entropy', help='Loss criterion')
    parser.add_argument('--learned_metric', type=str, default='epoch', help='Iteration Learned or Epoch Learned')

    parser.add_argument('--knn_k', default=30, type=int, help='k nearest neighbors of knn classifier')
    parser.add_argument('--seeds', nargs='+', type=int, default=[9203, 9304, 3456, 5210],
                        help='Seed values')
    return parser.parse_args()
