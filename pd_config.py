import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')

    parser = argparse.ArgumentParser(description='arguments to compute prediction depth for each data sample')
    parser.add_argument('--train_ratio', default=0.5, type=float, help='ratio of train split / total data split')
    parser.add_argument('--result_dir', default='./cl_results', type=str, help='directory to save ckpt and results')
    parser.add_argument('--model_dir', default='./pd_model', type=str, help='directory to save ckpt and results')
    parser.add_argument('--data', default='cifar10', type=str, help='dataset')
    parser.add_argument('--model', default='vgg', type=str, help='Model name')
    parser.add_argument('--get_train_pd', default=False, type=bool, help='get prediction depth for training split')
    parser.add_argument('--get_val_pd', default=True, type=bool, help='get prediction depth for validation split')
    parser.add_argument('--crit', type=str, default='cross_entropy', help='Loss criterion')
    parser.add_argument('--resume', default=False, type=bool, help='resume from the ckpt')
    parser.add_argument('--fraction', default=0.4, type=float, help='ratio of noise')
    parser.add_argument('--half', default=False, type=str,
                        help='use amp if GPU memory is 15 GB; set to False if GPU memory is 32 GB ')
    parser.add_argument('--num_epochs', default=80, type=int, help='number of epochs for training')
    parser.add_argument('--total_iteration', default=15000, type=str,
                        help='if training process is more than total iteration then stop')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--num_samples', default=10000, type=int, help='number of samples')
    parser.add_argument('--knn_k', default=30, type=int, help='k nearest neighbors of knn classifier')

    return parser.parse_args()
