import os
import time
sizes = [90,900,9000]
iterations = [1000, 5000, 10000]

gol_CPU = 'python GOL_CPU.py {} {}'
gol_GPU = 'python GOL_GPU.py {} {}'

for s in sizes:
    for i in iterations:
        print("CPU: ",str(os.popen(gol_CPU.format(s, i)).readlines()).split(
            " ")[3], ' --- ', s, i)
        time.sleep(0.3)
        print("GPU: ",str(os.popen(gol_GPU.format(s, i)).readlines()).split(
            " ")[3], ' --- ', s, i)
