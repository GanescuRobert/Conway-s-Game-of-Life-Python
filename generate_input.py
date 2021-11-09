import numpy as np

VALUES = [0,1]
RATIO =[[0.25,0.75],[0.5,0.5],[0.75,0.25]]
SIZE = [90,900,9000]

for r in RATIO:
    for s in SIZE:
        nums=np.random.choice(VALUES, size=s**2, p=r)
        with open (f'input_data/ratio_{r}___size_{s}x{s}.txt','w') as file:
            for num in nums:
                file.write(f'{num}')
    