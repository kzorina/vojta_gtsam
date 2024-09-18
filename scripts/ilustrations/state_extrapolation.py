import matplotlib.pyplot as plt
fig, ax = plt.subplots()
import numpy as np

t = np.array((0, 0))
v = np.array((20, 0))
a = np.array((10, -10))
Q = np.eye(2) * 0.1
Qv = np.eye(2) * 0.1
Qa = np.eye(2) * 0.1

for dt in range(3):
    ext_t = t + dt*v + a*dt*dt/2
    ext_Q = Q + Qv*dt**2 + Qa*dt**3/3
    ax.add_patch(plt.Circle(ext_t, 0.05, color='r'))
    print(ext_t, v + a*dt, np.linalg.norm(v + a*dt))

for dt in range(3):
    ext_t = t + dt*v + a*dt*dt/2
    ext_Q = Q + Qv*dt**2 + Qa*dt**3/3
    print(ext_Q)
ax.set_ylim(-3, 1)
ax.set_xlim(-1, 3)
plt.show()