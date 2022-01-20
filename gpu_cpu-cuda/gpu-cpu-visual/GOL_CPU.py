from scipy import signal
from utils import *
kernel = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
])

class GameOfLife:

    def __init__(self, workspace, shape, interval, num_frames):
        self.workspace = workspace
        self.shape = shape
        self.interval = interval 
        self.num_frames = num_frames
        self.image = None
        self.root = tk.Tk()
        self.root.title("Conway's Game of Life")
        self.frame = tk.Frame(master=self.root, width=shape[0], height=shape[1])
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame, width=shape[0], height=shape[1], highlightthickness=0)
        self.canvas.place(x=0, y=0)
        self.root.after(0, self.step)

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

    def update(self):
        count = signal.convolve2d(self.workspace.astype(int), kernel, mode='same')
        self.workspace &= (count == 2) 
        self.workspace |= (count == 3)

    def render(self):
        image = Image.fromarray(np.uint8(self.workspace).T * 0xff)
        self.image = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)
        self.root.update()

def main(shape,num_frames,workspace):
    app=GameOfLife(workspace=workspace,shape=shape,interval=0,num_frames=num_frames)
    app.start()
  
    
if __name__ == '__main__':
    size = int(sys.argv[1])
    num_frames = int(sys.argv[2])
    shape = (size,size)
    workspace = np.random.randint(0, 2, shape, dtype=bool)
    main(shape,num_frames,workspace)