import os

ratio = [0.25, 0.5, 0.75]
sizes = [90, 900, 9000]
iterations_default = [1000, 5000, 10000]
iterations_9000 = list(range(250, 1000+1, 250))

gol_secvential = 'python GOL_sequentially.py {} {} {}'
gol_parallel_liniar = 'mpiexec -n 9 python GOL_parallel_linearity.py {} {} {}'
gol_parallel_granularitate = 'mpiexec -n 9 python GOL_parallel_granularity(3x3).py {} {} {}'

for r in ratio:
    for s in sizes:
        # iterations = iterations_9000 if s == 9000 else iterations_default
        for i in iterations_default:
            #print( s, i, r)
            print(str(os.popen(gol_secvential.format(s, i, r)).readlines()).split(
                " ")[3], ' --- ', s, i, r)
