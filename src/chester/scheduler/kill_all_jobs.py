import os

from .. import config

for node in config.AUTOBOT_NODELIST:
    real_node = 'autobot-' + node
    print("killing ", real_node)
    command = f'ssh {real_node} "pkill -9 -u {config.SCHEDULER_USERNAME}"'

    os.system(command)
