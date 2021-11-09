# Conway-s-Game-of-Life-Python

Link to input data.
Google drive - https://drive.google.com/drive/folders/1YOsPhKZABWf83Cu4n12oZVdnTZkHCMfs?usp=sharing

References:
Parallelization: Conway’s Game of Life By Aaron	Weeden,	Shodor Education Foundation, Inc
http://www.shodor.org/media/content/petascale/materials/UPModules/GameOfLife/Life_Module_Document_pdf.pdf

https://github.com/thekartikay/game-of-life
https://github.com/ramizdundar/cmpe300-project
https://github.com/Jauntbox/paralife

CUDA In Your Python: Effective Parallel Programming on the GPU
https://www.youtube.com/watch?v=CO4ifMknS84
{

    CUDA C++ Programming Guide
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

    Cython to generate C++ class - Starting point
    https://github.com/rmcgibbo/npcuda-example
}

mpiexec -n 9 python GOL_parallel_granularity(3x3).py 90 100 0.5
main function took 2680.996 ms

mpiexec -n 9 python GOL_parallel_linearity.py 90 100 0.5        
main function took 1945.998 ms

python GOL_sequentially.py 90 100 0.5
main function took 1250.499 ms

