import glob
import sys
import os
import argparse

if __name__ == '__main__':
    batch_dir = sys.argv[1]
    is_dry = int(sys.argv[2])
    launch_scripts = glob.glob(batch_dir + "/*/slurm_launch")
    print(f"Found {len(launch_scripts)} launch scripts in {batch_dir}")
    for x in launch_scripts:
        print("Launching   {script}".format(script=x))
        if not is_dry:
            os.system("sbatch {script}".format(script=x))
