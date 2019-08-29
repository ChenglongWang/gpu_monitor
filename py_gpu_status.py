import os
import argparse
import subprocess

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    if args.mode == 'memory':
        option = 'memory.used,memory.total'
    elif args.mode == 'gpu':
        option = 'utilization.gpu'
    elif args.mode == 'all':
        option = 'memory.used,memory.total,utilization.gpu'
    
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu='+option,
            '--format=csv,nounits,noheader'
        ])

    if args.mode == 'memory':
        gpu_memory = [x.split(', ') for x in result.decode("utf-8").strip().split('\n')]
        gpu_memory = [int(x)/int(y)*100 for x, y in gpu_memory]
        print(','.join(map(str,gpu_memory)))
    elif args.mode == 'gpu':
        gpu_usage = [int(x) for x in result.decode("utf-8").strip().split('\n')]
        print(','.join(map(str,gpu_usage)))
    elif args.mode == 'all':
        gpu_status = [x.split(', ') for x in result.decode("utf-8").strip().split('\n')]
        gpu_memory = [int(x)/int(y)*100 for x, y, _ in gpu_status]
        gpu_usage  = [int(z) for _, _, z in gpu_status]
        print(','.join(map(str,gpu_memory)))
        print(','.join(map(str,gpu_usage)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='memory', help='Return gpu status flag')
    args = parser.parse_args()

    get_gpu_memory_map()
