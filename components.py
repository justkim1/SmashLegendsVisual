from PIL import Image
import numpy as np
from collections import deque

im = np.array(Image.open("reference/smash_reference.png")).astype(np.int16)
r, g, b = im[:,:,0], im[:,:,1], im[:,:,2]

mask = (r > 140) & (r > g * 1.35) & (r > b * 1.25)

h, w = mask.shape
seen = np.zeros_like(mask, dtype=bool)
components = []

for y in range(0, h, 3):
    for x in range(0, w, 3):
        if not mask[y, x] or seen[y, x]:
            continue

        q = [(y, x)]
        seen[y, x] = True
        count = 0
        minx = maxx = x
        miny = maxy = y

        while q:
            yy, xx = q.pop()
            count += 1
            minx = min(minx, xx)
            maxx = max(maxx, xx)
            miny = min(miny, yy)
            maxy = max(maxy, yy)

            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ny, nx = yy + dy, xx + dx

                if 0 <= ny < h and 0 <= nx < w:
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))

        if count >= 20:
            components.append((count, minx, miny, maxx, maxy))

components.sort(reverse=True)

print("TOTAL COMPONENTS:", len(components))
print()
print("TOP 30:")
for i, c in enumerate(components[:30], 1):
    count, x1, y1, x2, y2 = c
    print(f"{i:02d} pixels={count:6d} bbox=({x1},{y1})-({x2},{y2}) size={x2-x1+1}x{y2-y1+1}")
