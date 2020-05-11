import numpy as np
from skimage import color as clr

a = np.array([0.2722222222222222, 0.5, 0.73])
a = a.reshape(1, 1, -1)

# p, q, r = np.meshgrid(range(3), range(3), range(3), indexing='ij')
#
# a[0] * p.flatten()
# a[1] * q.flatten()
# a[2] * r.flatten()

a_rgb = clr.hsv2rgb(a)

b = np.concatenate((a, a_rgb), axis=None)
print(b)