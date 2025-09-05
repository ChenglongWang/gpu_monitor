import os
import argparse
import subprocess

def get_gpu_memory_map(mode='memory'):
    """Get the current gpu usage.

    Parameters
    ----------
    mode : str
        The mode to get GPU information ('memory', 'gpu', or 'all')

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    if mode == 'memory':
        option = 'memory.used,memory.total'
    elif mode == 'gpu':
        option = 'utilization.gpu'
    elif mode == 'all':
        option = 'memory.used,memory.total,utilization.gpu'
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'memory', 'gpu', or 'all'")
    
    try:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu='+option,
                '--format=csv,nounits,noheader'
            ], stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to run nvidia-smi: {e}")

    if mode == 'memory':
        gpu_memory = [x.split(', ') for x in result.decode("utf-8").strip().split('\n')]
        # Add division by zero protection
        gpu_memory = [int(x)/int(y)*100 if int(y) != 0 else 0 for x, y in gpu_memory]
        print(','.join(map(str,gpu_memory)))
    elif mode == 'gpu':
        gpu_usage = [int(x) for x in result.decode("utf-8").strip().split('\n')]
        print(','.join(map(str,gpu_usage)))
    elif mode == 'all':
        gpu_status = [x.split(', ') for x in result.decode("utf-8").strip().split('\n')]
        # Add division by zero protection
        gpu_memory = [int(x)/int(y)*100 if int(y) != 0 else 0 for x, y, _ in gpu_status]
        gpu_usage  = [int(z) for _, _, z in gpu_status]
        print(','.join(map(str,gpu_memory)))
        print(','.join(map(str,gpu_usage)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='memory', help='Return gpu status flag')
    args = parser.parse_args()

    get_gpu_memory_map(args.mode)
