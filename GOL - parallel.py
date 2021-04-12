import numpy as np
import time
import sys
from matplotlib import pyplot as plt
from mpi4py import MPI

def generate_images(i, result):
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
    each_image_duration = 1  # in secs
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # define the video codec

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape
    video = cv2.VideoWriter(video_name, fourcc, 1.0, (width, height))

    for image in images:
        for _ in range(each_image_duration):
            video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
status = MPI.Status()

COLS = ROWS = int(sys.argv[1])
generations = int(sys.argv[2])
ratio_negative = float(sys.argv[3])
distribution = ROWS//size

class Game(object):
    def __init__(self, shape, workspace, dtype=np.int8):
        self.workspace = np.ndarray(shape=shape, dtype=dtype,buffer=np.array(workspace))
        self.shape = self.workspace.shape
        self._engine = Engine(self)
        self.step = 0

    def animate(self,no_iter):
        while no_iter != self.step:
            self._engine._next_state()
            self.step += 1
class Engine(object):
    def __init__(self, workspace, dtype=np.int8):
        self._workspace = workspace
        self.shape = workspace.shape
        self.neighbor = np.zeros(workspace.shape, dtype=dtype)
        self._neighbor_id = self._make_neighbor_indices()
        
    def _make_neighbor_indices(self):
        d = [slice(None), slice(1, None), slice(0, -1)]
        d2 = [(0, 1), (1, 1), (1, 0), (1, -1)]
        out = [None for i in range(8)]
        for i, idx in enumerate(d2):
            x, y = idx
            out[i] = [d[x], d[y]]
            out[7 - i] = [d[-x], d[-y]]
        return out

    def _count_neighbors(self):
        self.neighbor[:, :] = 0  # reset neighbors
        # count #neighbors of each cell.
        w = self._workspace.workspace
        n_id = self._neighbor_id
        n = self.neighbor
        for i in range(8):
            n[tuple(n_id[i])] += w[tuple(n_id[7 - i])]
    
    def _up(self):
        w = self._workspace.workspace
        comm.send(w[distribution, :], dest=rank+1)
        w[distribution+1, :] = comm.recv(source=rank+1)
    
    def _down(self):
        w = self._workspace.workspace
        comm.send(w[1, :], dest=rank-1)
        w[0, :] = comm.recv(source=rank-1)
       
    def _set_ghost(self):
        if rank == 0:
            self._up()
        elif rank == size-1:
            self._down()
        else:
            self._up()
            self._down()   
    
    def _update_workspace(self):
        w = self._workspace.workspace
        n = self.neighbor
        w &= (n == 2) 
        w |= (n == 3)
        self._set_ghost()  

    def _next_state(self):
        self._count_neighbors()
        self._update_workspace()

def add_border(grid):
    zeroes_col = np.zeros((1, ROWS+2))
    zeroes_row =  np.zeros((distribution, 1))

    grid = np.hstack((grid, zeroes_row))
    grid = np.hstack((zeroes_row, grid))
    grid = np.vstack((grid, zeroes_col))
    grid = np.vstack((zeroes_col, grid))
    if rank == 0:
        grid[0, :] = 1
    return grid
def generate_subGrid(data):
    start = distribution * COLS * rank
    end = start + distribution * COLS
    grid = np.reshape(np.array(data[start:end]), (distribution, COLS))
    grid = add_border(grid)
    return grid

data = []
if rank == 0:
    with open(f'input_data/ratio_[{ratio_negative}, {1-ratio_negative}]___size_{ROWS}x{COLS}.txt', 'rb') as file:
        data = [int(_) for _ in list(file.read())]  

data = comm.bcast(data, root=0)
subGrid = generate_subGrid(data)

c,l=len(subGrid),len(subGrid[0])
time1 = time.time()
game = Game((c,l), subGrid)
game.animate(generations)
time2 = time.time()

grid = comm.gather(subGrid[1:distribution+1, :], root=0)
if rank == 0:
    print('function took {:.3f} ms'.format((time2-time1)*1000.0))
    result = np.vstack(grid)
    generate_images(0,result)

