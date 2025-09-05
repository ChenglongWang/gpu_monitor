# gpu_monitor
Simple python script for monitoring GPUs on remote computing servers.
Visdom is used as backend for visualization.

This script is used for our own purpose, not generally designed for other environments.

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start Visdom server:
```bash
python -m visdom.server
```

2. Run the GPU monitor:
```bash
python gpu_list.py --env GPUs
```

For standalone GPU status checking:
```bash
python py_gpu_status.py --mode memory  # Get memory usage
python py_gpu_status.py --mode gpu     # Get GPU utilization  
python py_gpu_status.py --mode all     # Get both
```

## Recent Improvements

- Fixed critical scoping issues that prevented module imports
- Added proper error handling for subprocess calls and file operations
- Added division by zero protection for memory calculations
- Improved robustness for missing dependencies
- Added input validation and fallback values

**Samples**
![sample](/gpu_monitor_visdom.png)