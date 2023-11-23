import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from Layers import Layers

def draw_rectangle(axis, color, centre_A: np.ndarray, dir_X_A: np.ndarray, dir_Y_A: np.ndarray, facecolor=(0, 0, 0, 0)):
    corners_A = [centre_A - dir_X_A / 2 - dir_Y_A / 2, centre_A - dir_X_A / 2 + dir_Y_A / 2,
                 centre_A + dir_X_A / 2 + dir_Y_A / 2, centre_A + dir_X_A / 2 - dir_Y_A / 2]
    rect_A = patches.Polygon(corners_A, linewidth=1, edgecolor=color, fc=facecolor)
    axis.add_patch(rect_A)


def random_rectangle(side_length):
    centre = np.random.rand(2)/10
    theta = np.random.rand(1)[0]
    a = np.array((side_length, 0))
    R = np.array(((np.cos(theta), -np.sin(theta)),
                  (np.sin(theta), np.cos(theta))))
    dir_X = np.dot(R, a)
    dir_Y = np.array((dir_X[1], -dir_X[0]))
    return centre, dir_X, dir_Y

def test_colisions():
    layers = Layers()
    for x in range(100):
        centre_A, dir_X_A, dir_Y_A = random_rectangle(0.05)
        centre_B, dir_X_B, dir_Y_B = random_rectangle(0.05)
        dir_X_A *= 2
        dir_X_B *= 2
        print(layers.rectangles_colide(centre_A, dir_X_A, dir_Y_A, centre_B, dir_X_B, dir_Y_B))
        fig, ax = plt.subplots()
        draw_rectangle(ax, 'r', centre_A, dir_X_A, dir_Y_A)
        draw_rectangle(ax, 'b', centre_B, dir_X_B, dir_Y_B)
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        plt.show()

def draw_layer(ax, layer, title = "-",  show_grips = False):
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    for key in layer:
        cube = layer[key]
        # a = cube.get_colision_rectangle()
        centre, _, _ = cube.get_collision_rectangle()
        if show_grips:
            for grip, color in zip(cube.grips, ['r', 'r', 'r', 'b', 'b', 'b']):
                draw_rectangle(ax, color, *grip.get_collision_rectangle())
        draw_rectangle(ax, 'g', *cube.get_collision_rectangle(), facecolor=(0, 1, 0, 0.5))
        ax.text(*(centre - np.array([0.005, 0.005])), cube.idx)
    ax.set_xlabel('X[m]')
    ax.set_ylabel('Y[m]')

def draw_layers(layers, show_grips = False):
    figure, axis = plt.subplots(1, len(layers.layers))
    for i in range(len(layers.layers)):
        draw_layer(axis[i], layers.layers[i], f"layer {i}", show_grips)
    plt.show()

def main():
    layers = Layers()
    layer = layers.get_random_layer(5, 0)
    # layer = layers.get_custom_layer(0)
    layers.layers[0] = layer
    print("colisions:")
    layers.update_colisions()
    layers.resolve_grabbability(0)
    draw_layer(layer)
    # test_colisions()


if __name__ == "__main__":
    main()