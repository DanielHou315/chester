import os
print(os.getcwd())
for i in range(12):
    os.system('sbatch chester/scripts/availability_test.sh')
