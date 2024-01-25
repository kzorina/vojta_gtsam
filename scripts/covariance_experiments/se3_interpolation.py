__author__ = "Mederic Fourmy"

import time
import numpy as np
import pinocchio as pin
import meshcat
import meshcat.geometry as g


scale_triad = 0.2


T1 = pin.SE3.Random()
T2 = pin.SE3.Random()

vis = meshcat.Visualizer()
vis.wait()

frame_axes = g.triad(scale_triad)
vis['T1'].set_object(frame_axes)
vis['T1'].set_transform(T1.homogeneous)
vis['T2'].set_object(frame_axes)
vis['T2'].set_transform(T2.homogeneous)

# Line between frame origins
line_vertices = np.array([
    T1.translation,
    T2.translation,
])
vis['contour_normals'].set_object(g.LineSegments(
    g.PointsGeometry(line_vertices.T)
))

frame_axes = g.triad(0.8*scale_triad)
vis['Tt'].set_object(frame_axes)


SLEEP = 0.01
NB_SAMPLES = 1000
t_arr = np.linspace(start=0, stop=1, num=NB_SAMPLES)

# input('Press key to start se(3) interpolation')
eta_12_se3 = pin.log6(T1.inverse() * T2)
# for t in t_arr:
#     Tt = T1 * pin.exp(eta_12_se3*t)
#     vis['Tt'].set_transform(Tt.homogeneous)
#     time.sleep(SLEEP)

# input('Press key to start t3xso(3) interpolation')
dp_12_t3se3 = T1.rotation.T @ (T2.translation - T1.translation)
do_12_t3se3 = pin.log3(T1.rotation.T @ T2.rotation)
# for t in t_arr:
#     Tt = pin.SE3.Identity()
#     Tt.translation = T1.translation + T1.rotation @ dp_12_t3se3*t
#     Tt.rotation = T1.rotation @ pin.exp3(do_12_t3se3*t)
#     vis['Tt'].set_transform(Tt.homogeneous)
#     time.sleep(SLEEP)
#
# input('Press key to start r3xso(3) interpolation')
dp_12_r3se3 = T2.translation - T1.translation
do_12_r3se3 = pin.log3(T1.rotation.T @ T2.rotation)
# for t in t_arr:
#     Tt = pin.SE3.Identity()
#     Tt.translation = T1.translation + dp_12_r3se3*t
#     Tt.rotation = T1.rotation @ pin.exp3(do_12_t3se3*t)
#     vis['Tt'].set_transform(Tt.homogeneous)
#     time.sleep(SLEEP)


# input('Press key to start r3xso(3) interpolation')
# T12 = T1.inverse() * T2
# my_dp_12_r3se3 = T12.translation
# my_do_12_r3se3 = pin.log3(T12.rotation)
# for t in t_arr:
#     Tt = pin.SE3.Identity()
#     Tt.translation = T1.translation + T1.rotation@dp_12_r3se3*t
#     Tt.rotation = T1.rotation @ pin.exp3(do_12_t3se3*t)
#     vis['Tt'].set_transform(Tt.homogeneous)
#     time.sleep(SLEEP)
input('Press key to start r3xso(3) interpolation')
T12 = T1.inverse() * T2
my_dp_12_r3se3 = T12.translation
my_do_12_r3se3 = pin.log3(T12.rotation)
Tt = T1
step = pin.exp(eta_12_se3/NB_SAMPLES)
for t in t_arr:
    Tt = Tt * step
    vis['Tt'].set_transform(Tt.homogeneous)
    time.sleep(SLEEP)
