
from mpi4py import MPI
import math
import numpy as np
import time
import sys

from generate_image_video import generate_image, generate_video


def is_perfect_square(x):
    return math.isqrt(x) ** 2 == x


def get_distribution(number):
    number /= 9
    if (int(number) != number or not is_perfect_square(int(number))):
        print("Error: Size of input is wrong.")
        exit()
    return int(number)


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
status = MPI.Status()

SIZE = int(sys.argv[1])
generations = int(sys.argv[2])
ratio = float(sys.argv[3])

# number of values for each process
distribution = get_distribution(SIZE**2)

# size_grid = number of values on rows or columns
size_grid = int(math.sqrt(distribution))

# shape of each grid
shape_grid = (size_grid, size_grid)
shape_grid_size = shape_grid[0]*shape_grid[1]


def timing(f):
    def wrap(*args, **kwargs):
        if rank == 0:
            time1 = time.time()
        ret = f(*args, **kwargs)
        if rank == 0:
            time2 = time.time()
            print('{:s} function took {:.3f} ms'.format(
                f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


class Game(object):
    def __init__(self,  workspace):
        self._workspace = workspace
        self._engine = Engine(self)

    def _animate(self, no_iter):
        while no_iter:
            self._gather_grid()
            self._engine._next_state()
            no_iter -= 1

    def _gather_grid(self):
        gatherGrid = comm.gather(self._workspace[1:-1, 1:-1], root=0)
        if rank == 0:
            gatherGrid = self._preprocess_grid(gatherGrid)
            generate_image("granularity_img", gatherGrid)

    def _preprocess_grid(self, gatherGrid):
        grid = []
        grid.append(np.hstack(gatherGrid[0:3]))
        grid.append(np.hstack(gatherGrid[3:6]))
        grid.append(np.hstack(gatherGrid[6:9]))
        return np.vstack(grid)


class Engine(object):
    def __init__(self, game, dtype=np.int16):
        self._game = game
        self._shape = game._workspace.shape
        self._neighbor = np.zeros(self._shape, dtype=dtype)
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
        self._neighbor[:, :] = 0
        w = self._game._workspace
        n_id = self._neighbor_id
        n = self._neighbor
        for i in range(8):
            n[tuple(n_id[i])] += w[tuple(n_id[7 - i])]

    def _up(self):
        w = self._game._workspace
        comm.send(w[1, :], dest=rank-3)
        w[0, :] = comm.recv(source=rank-3)

    def _down(self):
        w = self._game._workspace
        comm.send(w[-2, :], dest=rank+3)
        w[-1, :] = comm.recv(source=rank+3)

    def _right(self):
        w = self._game._workspace
        comm.send(w[:, -2], dest=rank+1)
        w[:, -1] = comm.recv(source=rank+1)

    def _left(self):
        w = self._game._workspace
        comm.send(w[:, 1], dest=rank-1)
        w[:, 0] = comm.recv(source=rank-1)

    def _left_down(self):
        w = self._game._workspace
        comm.send(w[-2, 1], dest=rank+2)
        w[-1, 0] = comm.recv(source=rank+2)

    def _left_up(self):
        w = self._game._workspace
        comm.send(w[1, 1], dest=rank-4)
        w[0, 0] = comm.recv(source=rank-4)

    def _right_down(self):
        w = self._game._workspace
        comm.send(w[-2, -2], dest=rank+4)
        w[-1, -1] = comm.recv(source=rank+4)

    def _right_up(self):
        w = self._game._workspace
        comm.send(w[1, -2], dest=rank-2)
        w[0, -1] = comm.recv(source=rank-2)

    def _lr_down(self):
        self._left_down()
        self._right_down()

    def _lr_up(self):
        self._left_up()
        self._right_up()

    def _lr_ud(self):
        self._lr_up()
        self._lr_down()

    def _l_ud(self):
        self._left_up()
        self._left_down()

    def _r_ud(self):
        self._right_up()
        self._right_down()

    def _set_ghost(self):
        eval({
            0: "self._right(), self._down(), self._right_down()",
            1: "self._right(), self._down(), self._left(), self._lr_down()",
            2: "self._down(), self._left(), self._left_down()",
            3: "self._up(), self._right(), self._down(), self._r_ud()",
            4: "self._up(), self._right(), self._down(), self._left(), self._lr_ud()",
            5: "self._up(), self._down(), self._left(), self._l_ud()",
            6: "self._up(), self._right(), self._right_up()",
            7: "self._up(), self._right(), self._left(), self._lr_up()",
            8: "self._up(), self._left(), self._left_up()"
        }[rank])

    def _update_workspace(self):
        w = self._game._workspace
        n = self._neighbor
        w &= (n == 2)
        w |= (n == 3)

    def _clear_border(self):
        w = self._game._workspace
        w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = 0

    def _next_state(self):
        self._clear_border()
        self._set_ghost()
        self._count_neighbors()
        self._update_workspace()


def add_border(grid):

    zeros = np.zeros(size_grid).reshape(-1, 1)
    grid = np.hstack((zeros, grid, zeros))

    zeros = np.zeros(size_grid+2)
    grid = np.vstack((zeros, grid, zeros))

    return grid


def generate_subgrid(data):
    start = shape_grid_size * rank
    end = start + shape_grid_size
    grid = np.reshape(data[start:end], shape_grid)
    grid = add_border(grid).astype(np.int16)
    return grid


def read_data():
    file_name = f'/ratio_[{ratio}, {1-ratio}]___size_{SIZE}x{SIZE}'
    with open('input_data/licenta'+file_name+'.txt', 'r') as file:
        return np.array([int(_) for _ in list(file.read())])


def distribute_data():
    data = []
    if rank == 0:
        data = read_data()
    data = comm.bcast(data, root=0)
    return generate_subgrid(data)


@timing
def main():
    subgrid = distribute_data()
    game = Game(subgrid)
    game._animate(generations)
    if rank == 0:
        generate_video("granularity")


if __name__ == "__main__":
    main()
