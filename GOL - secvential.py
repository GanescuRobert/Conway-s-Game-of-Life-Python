import numpy as np
import time
import sys

COLS = ROWS = int(sys.argv[1])
generations = int(sys.argv[2])
ratio_negative = float(sys.argv[3])

workspace_shape = (ROWS,COLS)
iteration_size = generations
with open(f'input_data/ratio_[{ratio_negative}, {1-ratio_negative}]___size_{ROWS}x{COLS}.txt', 'rb') as file:
        workspace =[int(_) for _ in list(file.read())]  
def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap

class Game(object):
    def __init__(self, shape, workspace, dtype=np.int8):
        self.workspace = np.ndarray(shape=shape, dtype=dtype,buffer=np.array(workspace))
        self.shape = self.workspace.shape
        self._engine = Engine(self)
        self.step = 0

    def animate(self,no_iter):
        while no_iter != self.step:
            self._engine.next_state()
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

    def _update_workspace(self):
        w = self._workspace.workspace
        n = self.neighbor
        w &= (n == 2) 
        w |= (n == 3)  

    def next_state(self):
        self._count_neighbors()
        self._update_workspace()
@timing
def main():
    game = Game(workspace_shape, workspace)
    game.animate(iteration_size)
main()