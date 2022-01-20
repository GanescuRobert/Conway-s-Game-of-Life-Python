from utils import *
from scipy import signal


kernel = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
])

class GameOfLife:
    def __init__(self, workspace, shape, num_frames):
        self.workspace = workspace
        self.shape = shape
        self.num_frames = num_frames
        
    def start(self):
        while self.num_frames:
            self.update() 
            self.num_frames-=1

    def update(self):
        count = signal.convolve2d(self.workspace.astype(int), kernel, mode='same')
        self.workspace = self.workspace & (count == 2) | (count == 3)

@timing
def main(shape,num_frames):
    GameOfLife(shape=shape,num_frames=num_frames).start()


if __name__ == '__main__':
    size = int(sys.argv[1])
    num_frames = int(sys.argv[2])
    shape = (size,size)
    workspace = np.random.randint(0, 2, shape, dtype=bool)
    main(shape,num_frames,workspace)
    