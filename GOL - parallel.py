import numpy
from matplotlib import pyplot as plt
from mpi4py import MPI
import datetime

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
status = MPI.Status()

t0=0

COLS = ROWS = 100
generations = 100000
distribution = ROWS//size

def up(grid):
    comm.send(grid[distribution-2, :], dest=rank+1)
    grid[distribution-1, :] = comm.recv(source=rank+1)
def down(grid):
    comm.send(grid[1, :], dest=rank-1)
    grid[0, :] = comm.recv(source=rank-1)
def count_neigbours(row, col, grid):
    return \
        (   grid[row-1, col-1]  + grid[row-1, col]  + grid[row-1, col+1] +
            grid[row, col-1]                        + grid[row, col+1] +
            grid[row+1, col-1]  + grid[row+1, col]  + grid[row+1, col+1]
         )
def update(grid):
    temp_grid = numpy.copy(grid)
    for row in range(1, distribution-1):
        for col in range(1, COLS-1):
            neighbour_sum = count_neigbours(row, col, grid)
            if grid[row, col] == 1:
                temp_grid[row, col] = (neighbour_sum in [2, 3])
            else:
                temp_grid[row, col] = (neighbour_sum == 3)
    grid = numpy.copy(temp_grid)
    return grid
def add_ghost(grid):
    if rank == 0:
        up(grid)
    elif rank == size-1:
        down(grid)
    else:
        up(grid)
        down(grid)
def generate_subGrid():
    input_data = []
    with open("input_data/ratio_[0.5, 0.5]___size_100x100.txt",'r') as file:
        input_data = [int(_) for _ in list(file.read())]

    start = distribution*COLS*rank
    end = start + distribution *COLS
    return numpy.reshape(numpy.array(input_data[start:end]), (distribution, COLS))
def generate_images(i,result):
    zeroes = ""
    if i < 1000:
        zeroes = "0"
    if i < 100:
        zeroes = "00"
    if i < 10:
        zeroes = "000"
    plt.imsave('temp/'+zeroes+str(i)+'.jpg', result)
def generate_video():
    import cv2
    import os

    image_folder = 'temp'
    video_name = 'video.avi'
    each_image_duration = 1 # in secs
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # define the video codec

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, fourcc, 1.0, (width, height))

    for image in images:
        for _ in range(each_image_duration):
            video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
subGrid = generate_subGrid()
if rank ==0:
    t0 = datetime.datetime.now()

zeroes_col = numpy.zeros((distribution,1))
zeroes_row = numpy.zeros((1,ROWS+2))

subGrid = numpy.hstack((subGrid,zeroes_col))
subGrid = numpy.hstack((zeroes_col,subGrid))
subGrid = numpy.vstack((subGrid,zeroes_row))
subGrid = numpy.vstack((zeroes_row,subGrid))

if rank == 0:
    subGrid[0, :] = 1

oldGrid = comm.gather(subGrid[1:distribution-1, :], root=0)
for i in range(1, generations):
    subGrid = update(subGrid)
    add_ghost(subGrid)

    newGrid = comm.gather(subGrid[1:distribution-1, :], root=0)

    if rank == 0:
        result = numpy.vstack(newGrid)
#         generate_images(i,result)

if rank == 0:
    t1=datetime.datetime.now()
    t2=t1-t0
    print(t2.microseconds/ 1000)
#     videomaker()
