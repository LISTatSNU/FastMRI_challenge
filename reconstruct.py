import argparse
from pathlib import Path
from utils.learning.test_part import forward

def parse():
    parser = argparse.ArgumentParser(description='Test Unet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_Unet', help='Name of network')
    parser.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/', help='Directory of test data')
    
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument("--input_key", type=str, default='image_input', help='Name of input key')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    
    # acc4
    args.data_path = args.path_data / "acc4" / "image"    
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc4"
    print(args.forward_dir)
    forward(args)
    
    # acc8
    args.data_path = args.path_data / "acc8" / "image"    
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc8"
    print(args.forward_dir)
    forward(args)

