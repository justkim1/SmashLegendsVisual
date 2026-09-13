from PIL import Image
import numpy as np

im = np.array(Image.open("reference/smash_reference.png")).astype(np.int16)
r, g, b = im[:,:,0], im[:,:,1], im[:,:,2]

red = (r > 140) & (r > g * 1.35) & (r > b * 1.25)

print("IMAGE:", im.shape[1], "x", im.shape[0])
print("RED:", int(red.sum()), "pixels")
print("RED %:", round(red.mean() * 100, 3))

ys, xs = np.where(red)
if len(xs):
    print("RED BBOX:", int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
else:
    print("RED BBOX: NONE")
