import numpy as np
import datetime

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

workspace_shape = (1000,1000)
iteration_size = 10000

with open("input_data/ratio_[0.5, 0.5]___size_1000x1000.txt","r") as file:
    workspace = [int(_) for _ in list(file.read())]

t0=datetime.datetime.now()
game = Game(workspace_shape, workspace)
game.animate(iteration_size)
t1=datetime.datetime.now()
t2=t1-t0
print(t2.microseconds/ 1000)

