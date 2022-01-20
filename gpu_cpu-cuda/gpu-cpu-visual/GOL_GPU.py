from numba import cuda
from utils import *
@cuda.jit
def update_cell(before, after):
    x, y = cuda.grid(2)
    count = 0
    for i in (-1, 0, 1):
        if x + i < 0 or x + i >= before.shape[0]:
            continue
        for j in (-1, 0, 1):
            if i == 0 and j == 0:
                continue
            if y + j < 0 or y + j >= before.shape[1]:
                continue
            if before[x + i, y + j]:
                count += 1
    after[x, y] = count in (2, 3) if before[x, y] else count == 3
    
class GameOfLife():

    def __init__(self, workspace, shape, interval, num_frames):
        self.workspace = workspace
        self.shape = shape
        self.num_frames = num_frames
        self.interval = interval  # ms
        self.image = None
        self.root = tk.Tk()
        self.root.title("Conway's Game of Life")
        self.frame = tk.Frame(master=self.root, width=shape[0], height=shape[1])
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame, width=shape[0], height=shape[1], highlightthickness=0)
        self.canvas.place(x=0, y=0)
        self.root.after(0, self.step)
        self.device_workspace = (
            cuda.to_device(self.workspace),
            cuda.to_device(np.zeros(self.shape, dtype=bool)),
        )
        
    def start(self):
        self.root.mainloop()
    
    def stop(self):
        self.root.quit()
    
    def step(self):
        if self.num_frames:
            self.update()
            self.render()
            self.root.after(self.interval, self.step)
        else:
            self.root.after(2000, self.stop)
        self.num_frames -= 1
    
    def render(self):
        image = Image.fromarray(np.uint8(self.workspace).T * 0xff)
        self.image = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)
        self.root.update()
        
    def update(self):
        update_cell[self.shape, 1](*self.device_workspace)
        self.workspace = self.device_workspace[1].copy_to_host()
        self.device_workspace = tuple(reversed(self.device_workspace))  # flip
    
def main(shape,num_frames,workspace):
    app=GameOfLife(workspace=workspace,shape=shape,interval=0,num_frames=num_frames)
    app.start()
  
    
if __name__ == '__main__':
    size = int(sys.argv[1])
    num_frames = int(sys.argv[2])
    shape = (size,size)
    workspace = np.random.randint(0, 2, shape, dtype=bool)
    main(shape,num_frames,workspace)

    
    
    