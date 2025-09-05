#!/usr/bin/env python
# coding: utf-8


import os, sys, time, subprocess, re
import numpy as np
import pickle
from os.path import join, isfile, isdir
from collections import deque

from visdom import Visdom
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from subprocess import STDOUT, check_output
import HTML

# In[12]
def filter_num(line):
    return re.findall(r"[-+]?\d*\.\d+|\d+", line)

def ssh_host(hosts, mode='memory', num_gpus=4, verbose=False, script_path=None):
    """
    SSH to hosts and get GPU information
    
    Parameters
    ----------
    hosts : list
        List of hostnames to connect to
    mode : str
        Mode for GPU information ('memory', 'gpu', 'all')
    num_gpus : int
        Expected number of GPUs per host
    verbose : bool
        Whether to print verbose output
    script_path : str
        Path to the py_gpu_status.py script
    
    Returns
    -------
    tuple or list
        GPU information depending on mode
    """
    if script_path is None:
        script_path = os.path.dirname(os.path.abspath(__file__))
    
    info, info2, info3 = [], [], []
    for host in hosts:
        if mode != 'all':
            COMMAND=f"python3 {script_path}/py_gpu_status.py --mode {mode}"
        else:
            COMMAND = f'''
            python3 {script_path}/py_gpu_status.py --mode all
            free -th
            '''

        try:
            result = check_output(["ssh", host, COMMAND],
                                   stderr=STDOUT, 
                                   timeout=10)
        except Exception as e:
            print('Catch exception when ssh {}! Skip! {}'.format(host,e.args))
            info.append([-1]*num_gpus)
            if mode == 'all':
                info2.append([-1]*num_gpus)
                info3.append([-1]*num_gpus)
        else:
            if mode != 'all':
                result_ = result.decode("utf-8")
                gpu_memory = list(map(float,result_.strip('\n').split(',')))
                if len(gpu_memory) < num_gpus:
                    gpu_memory.extend([-1,]*(num_gpus-len(gpu_memory)))
                info.append(gpu_memory)
            else:
                result_ = result.decode("utf-8").strip('\n').split('\n')
                gpu_memory = list(map(float,result_[0].split(',')))
                gpu_usage  = list(map(float,result_[1].split(',')))
                try:
                    cpu_memory = list(map(float,filter_num(result_[3])))
                except (IndexError, ValueError):
                    # Handle case where free command output is not as expected
                    cpu_memory = [-1, -1, -1]
                
                if len(gpu_memory) < num_gpus:
                    gpu_memory.extend([-1,]*(num_gpus-len(gpu_memory)))
                if len(gpu_usage) < num_gpus:
                    gpu_usage.extend([-1,]*(num_gpus-len(gpu_usage)))
                info.append(gpu_memory)
                info2.append(gpu_usage)
                info3.append(cpu_memory)
                
    if mode == 'all':
        return info, info2, info3
    else:
        return info

def get_gpu_health(mem, use, th=[1,15]):
    if mem<1 and use<=th[0]:
        return 'Empty'
    elif mem>1 and use<th[0]:
        return 'Error?'
    elif mem>1 and th[0]<=use<=th[1]:
        return 'Unhealthy'
    elif mem>1 and use>th[1]:
        return 'Healthy'
    else:
        #print(f'Confused: mem-{mem},use-{use}')
        return 'Error?'

def get_tablerow(host_name, status, color_map=None):
    """
    Create a table row with status and colors
    
    Parameters
    ----------
    host_name : str
        Name of the host
    status : list
        List of status strings
    color_map : dict
        Mapping of status to colors
        
    Returns
    -------
    list
        Row data with host name and colored status items
    """
    if color_map is None:
        # Default color map - will be overridden by global color_map2 if available
        color_map = {'Empty': 'white', 'Error?': 'red', 'Healthy': 'lime', 'Unhealthy': 'yellow'}
    
    items = []
    for s in status:
        try:
            # Try to use global color_map2 first, then fallback to parameter
            color = globals().get('color_map2', color_map).get(s, 'white')
        except (AttributeError, TypeError):
            color = 'white'
        items.append([s, color])
    return [host_name, ] + items

class GPU_logger():
    def __init__(self,host_names,time_interval,time_range, 
                 show_summary=True, show_monitor=True, show_cpu_mem=False,
                 port=8098, env_name='main', verbose=False, 
                 script_path=None, home_path=None):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.hosts = host_names
        self.sleep = time_interval
        self.range = time_range
        self.show_summary = show_summary
        self.show_monitor = show_monitor
        self.show_cpu_mem = show_cpu_mem
        self.verbose=verbose
        self.win = {}
        self.num_gpus = 4
        self.max_length = self.range // self.sleep
        self.tick_ds = self.max_length//6 if self.max_length > 0 else 1
        self.time_indices = deque()
        self.memory_queue = np.zeros([len(host_names), self.num_gpus, 0])
        self.usages_queue = np.zeros([len(host_names), self.num_gpus, 0])
        self.table_W = 780
        self.table_H = 470
        
        # Set paths with fallbacks
        self.script_path = script_path if script_path else os.path.dirname(os.path.abspath(__file__))
        self.home_path = home_path if home_path else os.getenv('HOME', '/tmp')
        
        self.restore(self.script_path)

    def reset(self):
        self.viz.close(win=None, env=self.env)

    def save(self, output_dir):
        if not isdir(output_dir):
            os.makedirs(output_dir) 
        
        with open(join(output_dir, 'cached_data'), 'wb') as f:
            pickle.dump({'time': self.time_indices, 
                         'memory':self.memory_queue,
                         'usages':self.usages_queue}, f, pickle.HIGHEST_PROTOCOL)
    
    def restore(self, output_dir):
        if isfile(join(output_dir, 'cached_data')):
            with open(join(output_dir, 'cached_data'), 'rb') as f:
                cache = pickle.load(f)
                self.time_indices = cache['time']
                self.memory_queue = cache['memory']
                self.usages_queue = cache['usages']

    def record(self):
        if self.show_summary:
            gpu_tabel_opts = {'title':'README', 'resize':False, 'width':self.table_W, 'height':self.table_H}
            col_W = self.table_W/len(self.hosts)
            self.win['summary'] = self.viz.text(README, env=self.env, opts=gpu_tabel_opts)
        if self.show_cpu_mem:
            cpu_tabel_opts = {'title':'CPU Memory', 'resize':True, 'width':self.table_W, 'height':150}
            col_W_c = self.table_W/len(self.hosts)
            self.win['cpu_mem'] = self.viz.text('', env=self.env, opts=cpu_tabel_opts)
        
        while True:
            if len(self.time_indices) >= self.max_length:
                self.time_indices.popleft()
            self.time_indices.append(time.strftime("%H:%M"))

            # Safe file operation for tcshrc backup
            tcshrc_path = join(self.home_path, '.tcshrc')
            tcshrc_bak_path = join(self.home_path, '.tcshrc.bak')
            
            try:
                if isfile(tcshrc_path): # hotfix to handle my zsh
                    os.rename(tcshrc_path, tcshrc_bak_path)
            except (OSError, IOError) as e:
                if self.verbose:
                    print(f"Warning: Could not backup .tcshrc file: {e}")

            #gpu_memory = ssh_host(self.hosts, mode='memory')
            #gpu_usage  = ssh_host(self.hosts, mode='gpu')
            gpu_memory, gpu_usage, cpu_memory = ssh_host(self.hosts, mode='all', 
                                                          num_gpus=self.num_gpus, 
                                                          verbose=self.verbose, 
                                                          script_path=self.script_path)

            if self.verbose:
                print('gpu memory:', gpu_memory)
                print('gpu usage:', gpu_usage)
                print('cpu memory:', cpu_memory)

            try:
                if isfile(tcshrc_bak_path):
                    os.rename(tcshrc_bak_path, tcshrc_path)
            except (OSError, IOError) as e:
                if self.verbose:
                    print(f"Warning: Could not restore .tcshrc file: {e}")

            if self.memory_queue.shape[-1] < self.max_length:
                self.memory_queue = np.append(self.memory_queue, 
                                              np.reshape(gpu_memory,[len(self.hosts), self.num_gpus, 1]), 
                                              -1)
                self.usages_queue = np.append(self.usages_queue, 
                                              np.reshape(gpu_usage, [len(self.hosts), self.num_gpus, 1]), 
                                              -1)
            else:
                self.memory_queue = np.append(np.delete(self.memory_queue, 0, -1), 
                                              np.reshape(gpu_memory,[len(self.hosts), self.num_gpus, 1]), 
                                              -1)
                self.usages_queue = np.append(np.delete(self.usages_queue, 0, -1), 
                                              np.reshape(gpu_usage, [len(self.hosts), self.num_gpus, 1]),
                                              -1)
            gpu_status_table, cpu_status_table = [], []
            gpu_status_table.append(get_tablerow('',['GPU:01','GPU:02','GPU:03','GPU:04']))
            cpu_status_table.append(get_tablerow('',['Total:', 'Used:', 'Available:']))
            for k,host in enumerate(self.hosts):
                fig = plt.figure(figsize=(20, 2.7))
                fig.suptitle(host, size=24, fontweight="bold", y=1.)
                gpu_status_row = []
                for idx in range(4):
                    memory_, usage_ = self.memory_queue[k,idx,:], self.usages_queue[k,idx,:]
                    nonzero_idx = np.where(memory_>1)[0]
                    m_memory_ = np.mean(memory_[nonzero_idx[0]:]) if len(nonzero_idx) > 0 else np.mean(memory_)
                    m_usage_ = np.mean(usage_[nonzero_idx]) if len(nonzero_idx) > 0 else np.mean(usage_)
                    if m_memory_ == -1: # I've set empty gpu to -1
                        gpu_status_row.append(' ')
                        continue

                    if np.any(memory_[-5:]>1): #if last 5 checkpoint is empty -> not used now
                        status = get_gpu_health(m_memory_, m_usage_)
                    else:
                        status = status_kw[0]

                    box_prop = dict(facecolor=color_map[status], edgecolor='none', pad=2, alpha=0.6)
                    gpu_status_row.append(status)

                    plt.subplot(1,4,idx+1)
                    plt.xticks(np.arange(0, len(self.time_indices), self.tick_ds))
                    plt.fill_between(list(self.time_indices), memory_, color='r', label="Memory", alpha=0.4)
                    plt.fill_between(list(self.time_indices), usage_,  color='g', label="Usages", alpha=0.4)
                    plt.ylim(-1, 101)
                    plt.title('GPU-{:02d}'.format(idx),size=15,bbox=box_prop)
                    plt.xlabel('Average Memory: {:.1f}% Average Usage: {:.1f}%'\
                                .format(m_memory_, m_usage_), size=14)
                    plt.tight_layout()
                    plt.legend()

                gpu_status_table.append(get_tablerow(host.replace('emon','-'),gpu_status_row)) #shorten hostname
                cpu_status_table.append(get_tablerow(host.replace('emon','-'),
                                        [str(cpu_memory[k][0])+'G',str(cpu_memory[k][1])+'G',str(cpu_memory[k][-1])+'G']))
                if self.show_monitor:
                    opts = {'title':host, 'resizable':True, 'height':190, 'width':1400}
                    if host not in self.win.keys():
                        self.win[host] = self.viz.matplot(plt, env=self.env, opts=opts)
                    else:
                        self.viz.matplot(plt, win=self.win[host], env=self.env, opts=opts)
                plt.close()
            
            TIMESTAMP=f"Updated at {time.strftime('%Y/%m/%d-%H:%M:%S')}<br>"
            if 'summary' in self.win.keys():
                status_table_g = [list(t) for t in zip(*gpu_status_table)]
                table_ = HTML.table(status_table_g,border='3',cellpadding=5,
                                    col_width=[col_W]*(len(self.hosts)+1),
                                    col_align=['center']*(len(self.hosts)+1))
                
                self.viz.text(README+table_+TIMESTAMP+ACKNOWLEDGE,
                              env=self.env,win=self.win['summary'],
                              opts=gpu_tabel_opts)
            if 'cpu_mem' in self.win.keys():
                status_table_c = [list(t) for t in zip(*cpu_status_table)]
                table_c = HTML.table(status_table_c,border='2',cellpadding=2,
                                    col_width=[col_W_c]*(len(self.hosts)+1),
                                    col_align=['center']*(len(self.hosts)+1))
                self.viz.text(table_c+TIMESTAMP,
                              env=self.env,win=self.win['cpu_mem'],
                              opts=cpu_tabel_opts)
            time.sleep(self.sleep)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='GPUs', help='Env name')
    parser.add_argument('-i', '--interval', type=int, default=60, help='Time interval for monitoring (sec.)')
    parser.add_argument('-r', '--time-range', type=int, default=10800, help='Time range for showing data (sec.)')

    args = parser.parse_args()
    
    HOME = os.getenv('HOME')
    PWD = os.path.dirname(os.path.abspath(__file__))
    HOSTS = ['doraemon{:02d}'.format(i) for i in range(1,13)]
    status_kw = ['Empty','Error?','Healthy','Unhealthy']
    color_map  = {status_kw[0]:'none', status_kw[1]:'red',status_kw[2]:'green',status_kw[3]:'yellow'}
    color_map2 = {status_kw[0]:'white',status_kw[1]:'red',status_kw[2]:'lime', status_kw[3]:'yellow'}
    interval = args.interval #second
    time_range = args.time_range #second
    s = '&nbsp;'*4
    README = f"""
        こんにちは！より効率的に計算サーバーを使うために、このGPUモニタリングを作りました。<br><br>
        現在約{interval/60:0.1f}分ごとに更新します。表示範囲は約{time_range/3600:0.1f}時間です。<br>
        <b>Memory</b>: GPUメモリ使用量, <b>Usage</b>:GPU計算使用率<br>
        <h4>そして、ご参考にGPU使用状況を簡単に分析しました。リソースの無駄遣いはダメです&#x1f641;</h4>
        <font color="black">{s}<b>{status_kw[0]}</b></font>: 使用していません<br>
        <font color="lime">{s}<b>{status_kw[2]}</b></font>: (usage>15%) 効率的に使ってます！&#128077;<br>
        <font color="gold">{s}<b>{status_kw[3]}</b></font>: (usage<15%) 計算の効率が悪い, プログラムの改善をおすすめ&#x1f914;<br>
        <font color="red">{s}<b>{status_kw[1]}</b></font>: (usage&asymp;0%) メモリは使ってるが、GPUは動いてない。ジョブのチェックお願いします&#x1f64f;<br>
        <br>
    """
    ACKNOWLEDGE = f"""
        <br> PS1: You can focus on single manchine using <i>Filter text</i>.
        <br> PS2: Please dont change env <b>'GPUs'</b> in Environment setting!
        <br> PS3: If you accidently closed the window, it will appear after {interval} seconds.
        <br><br> Driven by <a href='https://ai.facebook.com/tools/visdom/'><b>Visdom</b></a>, coded by <b>CL.Wang</b>
    """

    logger = GPU_logger(HOSTS, interval, time_range, 
                        show_summary=True,
                        show_monitor=True,
                        show_cpu_mem=True,
                        env_name=args.env, verbose=False,
                        script_path=PWD, home_path=HOME)

    try:
        logger.record()
    except KeyboardInterrupt:
        logger.save(logger.script_path)
        logger.reset()
    except Exception as e:
        logger.save(logger.script_path)
        logger.reset()
        #exc_type, exc_obj, exc_tb = sys.exc_info()
        logger.viz.text(f"GPU monitor crashed at \
                         {time.strftime('%Y/%m/%d-%H:%M:%S')}!\n \
                         {e}", env=logger.env)
