import numpy as np
import time
import sys
from generate_image_video import generate_image, generate_video


SIZE = int(sys.argv[1])
generations = int(sys.argv[2])
ratio = float(sys.argv[3])

def read_data():
    file_name = f'/ratio_[{ratio}, {1-ratio}]___size_{SIZE}x{SIZE}'
    with open('input_data/licenta'+file_name+'.txt', 'r') as file:
        return np.array([int(_) for _ in list(file.read())])


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(
            f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


class Game(object):
    def __init__(self, workspace):
        self._workspace = workspace
        self._engine = Engine(self)

    def animate(self, no_iter):
        while no_iter:
            self._engine.next_state()
            no_iter -= 1


class Engine(object):
    def __init__(self, game):
        self._game = game
        self.neighbor = np.zeros(
            self._game._workspace.shape, dtype=np.int8)
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
        w = self._game._workspace
        n_id = self._neighbor_id
        n = self.neighbor
        for i in range(8):
            n[tuple(n_id[i])] += w[tuple(n_id[7 - i])]

    def _update_workspace(self):
        w = self._game._workspace
        n = self.neighbor
        w &= (n == 2)
        w |= (n == 3)

    def next_state(self):
        #!
        generate_image('sequentially_img', self._game._workspace)
        self._count_neighbors()
        self._update_workspace()


@timing
def main():
    data = read_data()
    data = np.reshape(data,(SIZE,SIZE)).astype(np.int16)
    game = Game(data)
    game.animate(generations)
    #!
    generate_video("sequentially")


if __name__ == "__main__":
    main()