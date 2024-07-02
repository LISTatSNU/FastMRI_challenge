import argparse
from pathlib import Path
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.test_part import forward
import time
    
def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/', help='Directory of test data')
    
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    
    public_acc, private_acc = None, None

    assert(len(os.listdir(args.path_data)) == 2)

    for acc in os.listdir(args.path_data):
      if acc in ['acc4', 'acc5', 'acc8']:
        public_acc = acc
      else:
        private_acc = acc
        
    assert(None not in [public_acc, private_acc])
    
    start_time = time.time()
    
    # Public Acceleration
    args.data_path = args.path_data / public_acc # / "kspace"    
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / 'public'
    print(f'Saved into {args.forward_dir}')
    forward(args)
    
    # Private Acceleration
    args.data_path = args.path_data / private_acc # / "kspace"    
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / 'private'
    print(f'Saved into {args.forward_dir}')
    forward(args)
    
    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')
    
    print('Success!') if reconstructions_time < 3000 else print('Fail!')