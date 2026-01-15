import os

from chester.autobot_config import AUTOBOT_NODELIST

for node in AUTOBOT_NODELIST:
    real_node = 'autobot-' + node
    print("killing ", real_node)
    command = f'ssh {real_node} "pkill -9 -u zixuanhu"'

    os.system(command)
