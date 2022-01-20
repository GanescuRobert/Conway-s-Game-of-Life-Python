from utils import *
from numba import cuda

@cuda.jit
def update_cell(before, after):
    x, y = cuda.grid(2)
    count = 0
    for i in (-1, 0, 1):
        # first ans last row
        if x + i < 0 or x + i >= before.shape[0]:
            continue
        for j in (-1, 0, 1):
            #center position
            if i == 0 and j == 0:
                continue
            # first ans last column
            if y + j < 0 or y + j >= before.shape[1]:
                continue
            if before[x + i, y + j]:
                count += 1
    after[x, y] = count in (2, 3) if before[x, y] else count == 3

class GameOfLife():

    def __init__(self, workspace, shape, num_frames):

        self.workspace = workspace
        self.shape = shape
        self.num_frames = num_frames
        
        self.device_workspace = (
            cuda.to_device(self.workspace),
            cuda.to_device(np.zeros(self.shape, dtype=bool)),
        )
        
    def start(self):
        while self.num_frames:
            self.update() 
            self.num_frames-=1
        
    def update(self):
        update_cell[self.shape, 1](*self.device_workspace)
        self.workspace = self.device_workspace[1].copy_to_host()
        self.device_workspace = tuple(reversed(self.device_workspace))
        
    
@timing
def main(size,num_frames,workspace):
    GameOfLife(workspace=workspace,shape=shape,num_frames=num_frames).start()


if __name__ == '__main__':
    size = int(sys.argv[1])
    num_frames = int(sys.argv[2])
    shape = (size,size)
    workspace = np.random.randint(0, 2, shape, dtype=bool)
    main(shape,num_frames,workspace)

    
    
    