# The scheduler sweeps over available GPUs for the nodes  in the CHESTER_QUEUE_DIR
import os
import glob
import pdb
import pickle
import time
import sys

from .. import config
from ..utils_logger import timelog
import psutil

check_interval = 120  # Check every 120 seconds for available GPUs


def checkIfProcessRunning(processName):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    scheduler_user = config.SCHEDULER_USERNAME
    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower() and scheduler_user == proc.username():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def check_available_nodes():
    all_stats_path = glob.glob(config.GPU_STATE_DIR + '/*.pkl')
    stats_names = [p.split('/')[-1][:-4] for p in all_stats_path]
    all_stats = []
    for p in all_stats_path:
        while 1:
            try:

                all_stats.append(pickle.load(open(p, 'rb')))
                time.sleep(0.1)
            except:
                print(p, ' is broken')
                continue
            break
    # all_stats = [ for p in all_stats_path]
    available_nodes = {}
    for name, (sys_data, gpu_data) in zip(stats_names, all_stats):
        gpu_ids = []
        for i, data in enumerate(gpu_data):
            if len(data['procs']) == 0:  # GPU is available if no user process is found on the GPU
                gpu_ids.append((i, data['mem_free']))
        if len(gpu_ids) > 0:
            available_nodes[name] = gpu_ids
    return available_nodes


if checkIfProcessRunning('remote_scheduler'):
    exit()

if __name__ == '__main__':
    print('cur folder ', os.getcwd())
    while 1:
        # check jobs in the queue
        tasks = glob.glob(os.path.join(config.CHESTER_QUEUE_DIR, 'queues', '*'))
        print(f' {len(tasks)} jobs to be scheduled')
        tasks_with_time = [(os.path.getmtime(t), t) for t in tasks if os.path.isfile(t)]
        sorted_tasks = sorted(tasks_with_time)
        # check if any GPUs are available
        available_GPUs = check_available_nodes()  # Dictionary: {node_name:[available_gpu_id],...]
        # available_GPUs.pop('autobot-0-17', None)
        # available_GPUs.pop('autobot-0-25', None)
        # available_GPUs.pop('autobot-0-33', None)
        # available_GPUs.pop('autobot-0-37', None)
        # available_GPUs.pop('autobot-0-23', None)
        # available_GPUs["autobot-1-1"] = [(0, 10000), (1, 10000),
        #                                  (2, 10000), (3, 10000)]
        short_list = [
            # "autobot-0-25",
            "autobot-0-29",
            "autobot-0-33",
            "autobot-0-37",
        ]
        # for node in short_list:
        #     if node in available_GPUs:
        #         available_GPUs[node] = available_GPUs[node][:6]
        short_list = [
            # "autobot-0-25",
            'autobot-0-9',
            'autobot-0-11',
            "autobot-0-13", "autobot-0-15", "autobot-0-17", "autobot-0-19", "autobot-0-21",
            'autobot-0-23',  # 4x2080 + 32 cpus
        ]
        for node in short_list:
            if node in available_GPUs:
                available_GPUs[node] = available_GPUs[node][:3]
        short_list = [
            "autobot-1-1", "autobot-1-6",
        ]
        for node in short_list:
            if node in available_GPUs:
                available_GPUs[node] = available_GPUs[node][:4]
        # available_GPUs = {
        #     # 'autobot-0-9': [0, 1],
        #     # 'autobot-0-11': [0, 1],
        #     # 'autobot-0-13': [0, 1],
        #     # 'autobot-0-15': [0, 1, 2, 3],
        #     'autobot-0-15': [0, 1],
        #     # 'autobot-0-17': [0, 1, 2, 3],
        #     'autobot-0-19': [0, 1],
        #     'autobot-0-21': [0, 1],
        #     # 'autobot-0-23': [0, 1],
        #     # 'autobot-0-25': [ 1, 2, 3, 4, 5, 6, 7],
        #     # 'autobot-0-29': [0, 1, 2, 3, 4, 5, 6, 7],
        #
        #     'autobot-1-1': [0, 1],
        #     'autobot-1-6': [0, 1],
        #     'autobot-1-10': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        # }  # Temporary
        # timelog('Available GPUs: ' + str(available_GPUs))
        for node, gpus in available_GPUs.items():
            print(f"##### {node}")
            for i, mem in gpus:
                print(f'    GPU {i}: {mem} MB')

        succ_tasks = 0
        for _, script in sorted_tasks:
            with open(script) as f:
                while True:  # Read header files
                    line = f.readline()
                    if line[0] != '#':
                        break
                    header = line.split(' ')[0][1:]
                    if header == 'CHESTERNODE':
                        node_list = line.rstrip()[13:].split(',')
                    elif header == 'CHESTEROUT':
                        stdout_file = line.rstrip().split(' ')[1]
                    elif header == 'CHESTERERR':
                        stderr_file = line.rstrip().split(' ')[1]
                    elif header == 'CHESTERSCRIPT':
                        script_file = line.rstrip().split(' ')[1]
            for node in node_list:
                real_node = 'autobot-' + node
                # TODO: add gpu num
                if real_node in available_GPUs:
                    gpu_ids = available_GPUs[real_node]
                    if len(gpu_ids) > 0:
                        env_command = f'CUDA_VISIBLE_DEVICES={gpu_ids[0][0]} '
                        command = f"ssh -q {real_node} \'{env_command} bash {script_file} </dev/null >{stdout_file} 2>{stderr_file} &\'"
                        rm_command = f'rm {script}'
                        timelog(f"Job launched on node {real_node}, GPU {gpu_ids[0]}, {script_file}")
                        os.system(command)
                        os.system(rm_command)
                        gpu_ids.pop(0)
                        succ_tasks += 1
                        break
                    else:
                        del available_GPUs[real_node]
        if succ_tasks == len(sorted_tasks):
            timelog(f'All {succ_tasks} jobs done!')
            print("==================== ", flush=True)
            if config.CHESTER_SCHEDULER_LOG_DIR:
                log_pattern = os.path.join(config.CHESTER_SCHEDULER_LOG_DIR, '*')
                log_files = sorted(glob.glob(log_pattern))
                if log_files:
                    last_log = log_files[-1]
                    dir_name = os.path.dirname(last_log)
                    os.system(f'cp {last_log}  {dir_name}/last_log.txt')
            break
        time.sleep(check_interval)
