import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def create_arg_parser():
    parser = argparse.ArgumentParser()
    
    # 数据路径参数
    parser.add_argument('--data_root', type=str, default=None,
                        help='root path of data')
    parser.add_argument('--data_list_file', type=str, default='./data/train.txt',
                        help='training data list file')
    parser.add_argument('--data_list_file_val', type=str, default='./data/val.txt',
                        help='validation data list file')
    
    parser.add_argument('--val_fold',
                        default=0, type=int, help='')
    
    
    # 模型相关参数
    parser.add_argument('--modality', default=['CT'], type=arg_as_list, help='CT, pathology, clinical info (CI)')
    parser.add_argument('--model_CT', default='resnetMC3_18_wMask', type=str) # resnetMC3_18, medicalNet
    parser.add_argument('--aggregator', default='TransMIL_seperate', type=str)
    parser.add_argument('--alignment_base', default='CT', type=str) # CT, pathology
    
    # training parameters
    parser.add_argument('--start_epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
    parser.add_argument('--batch_size', default=8, type=int, help='Mini batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of jobs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Strength of weight decay regularization')
    parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    
    # distributed training parameters
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=str,
                        help='GPU id to use.')
    parser.add_argument('--workers', default=16, type=int,
                        help='number of data loading workers')
    
    # survival analysis parameters
    parser.add_argument('--survival_type',
                        default='OS',
                        help='OS / RFS', type=str)
    
    # other parameters
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")
    parser.add_argument('--schedule', default=[50, 250, 500, 750], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')


    parser.add_argument('--clinical_features',
                        default=['sex', 'age', 'HBV', 'AFP', 'albumin', 'bilirubin', 'child', 'BCLC', 'numoflesions', 'diameter', 'VI'],
                        type=arg_as_list, help='Clinical features used for training')


    args = parser.parse_args()
    return args