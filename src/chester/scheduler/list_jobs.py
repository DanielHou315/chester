import os
from .. import config
from ..utils_logger import timelog
import psutil
import os
import glob
import pickle
import time


def check_available_nodes():
    all_stats_path = glob.glob(config.GPU_STATE_DIR + '/*.pkl')
    all_stats_path.sort()
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
    available_nodes = {}
    print("    Num process |   Node name")
    for name, (sys_data, gpu_data) in zip(stats_names, all_stats):
        counter = 0
        for i, data in enumerate(gpu_data):
            for proc_data in data['procs']:
                if proc_data[1] == 'zixuanhu':
                    counter += 1
        print(f"{counter:15} | {name}")
    return available_nodes


if __name__ == '__main__':
    check_available_nodes()
