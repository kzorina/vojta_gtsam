import numpy as np
import pinocchio as pin
from Cube import Cube
from typing import List, Dict, Set
from scipy.spatial.transform import Rotation

class Layers():
    def __init__(self):
        self.cube_side = 0.05  # in meters
        self.T_bt = np.array([[1, 0, 0, 0],  # table to robot base transformation
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        self.T_tb = np.linalg.inv(self.T_bt)
        self.layers:List[Dict[str:Cube]] = [{}, {}, {}, {}]

    def __get_cube_layer(self, T_bo:np.ndarray) -> int:
        """
        :param T_bo: cube to robot base.
        :return: the layer to which the cube belongs according to its height above the table.
        """
        T_to = self.T_tb @ T_bo
        z = T_to[2, 3]
        z_in_cubes = z/self.cube_side
        layer = round(z_in_cubes) - 1
        if layer < 0:
            layer = 0
        if layer > (len(self.layers) - 1):
            layer = len(self.layers) - 1
        return layer

    def __align_cube_with_layer(self, T_bo: np.ndarray, layer: int) -> np.ndarray:
        """
        :param T_bo: cube to robot base transformation.
        :param layer: layer with which the cube will be aligned.
        :return: cube to table transformation. The cube will be rotated and translated so that it is aligned with the table.
        """
        T_to = self.T_tb @ T_bo
        T_to[2, 3] = (layer + 1) * self.cube_side
        w = pin.log3(T_to[:3, :3])
        aligned_w = np.array((0, 0, 1)) * np.linalg.norm(w)
        T_to[:3, :3] = pin.exp3(aligned_w)
        return T_to

    def insert_cube(self, T_bo_estimate: np.ndarray, idx: str):
        """
        :param T_bo_estimate: cube to robot base transformation. Doesn't have to be aligned with the table.
        :param idx: preferably AruCo id. Has to be unique for each layer.
        """
        layer = self.__get_cube_layer(T_bo_estimate)
        for l in range(layer + 1):
            T_to = self.__align_cube_with_layer(T_bo_estimate, l)
            new_cube = Cube(idx, T_to, cube_side=self.cube_side)
            self.layers[l][idx] = new_cube


    def get_cube(self, idx:str):
        for layer in self.layers:
            if idx in layer:
                return layer[idx]
        return None

    def update_colisions(self):
        """
        Check every grip of every cube and detects all colisions. These colisions are saved in the Grip objects.
        """
        for layer in self.layers:
            for key in layer:
                cube: Cube = layer[key]
                other_cubes: List[Cube] = [c for k, c in layer.items() if k != key]
                for grip in cube.grips:
                    grip.colisions = []
                    for other_cube in other_cubes:
                        if self.rectangles_colide(*grip.get_collision_rectangle(),
                                                  *other_cube.get_collision_rectangle()):
                            grip.colisions.append(other_cube)
                            print(cube.idx, other_cube.idx)

    def get_random_layer(self, count, layer_id):
        layer = {}
        for i in range(count):
            t = (np.random.rand(3) - 0.5)/2
            t[2] = (layer_id + 1)*self.cube_side
            T_bo = np.zeros((4, 4), np.float64)
            T_bo[:3, :3] = Rotation.random().as_matrix()
            T_bo[:3, 3] = t
            T_bo[3, 3] = 1
            T_to = self.__align_cube_with_layer(T_bo, layer=layer_id)
            id = str(i)
            cube = Cube(id, T_to, self.cube_side)
            layer[id] = cube
        return layer

    def get_custom_layer(self, layer_id):
        layer = {}
        # positions = [np.array((0, 0, 0)), np.array((0.006, 0.006, 0)), np.array((0.006, 0, 0)), np.array((0, 0.006, 0)),
        #              np.array((0, 0.030, 0)), np.array((0.006, 0.030, 0)), np.array((0.00, 0.036, 0))]
        positions = [np.array((-0.01, 0, 0)), np.array((0.11, 0.0, 0)), np.array((0.05, 0.04, 0)), np.array((0.05, -0.02, 0))]
        for i, t in enumerate(positions):
            t[2] = (layer_id + 1)*self.cube_side
            T_bo = np.zeros((4, 4), np.float64)
            T_bo[:3, :3] = np.eye(3)
            T_bo[:3, 3] = t
            T_bo[3, 3] = 1
            T_to = self.__align_cube_with_layer(T_bo, layer=layer_id)
            id = str(i)
            cube = Cube(id, T_to, self.cube_side)
            layer[id] = cube
        return layer

    def rectangles_colide(self,
                          centre_A: np.ndarray,
                          dir_X_A: np.ndarray,
                          dir_Y_A: np.ndarray,
                          centre_B: np.ndarray,
                          dir_X_B: np.ndarray,
                          dir_Y_B: np.ndarray) -> bool:
        """
        Determines whether two rectangles colide.
        :param centre_A: a vector corresponding to the centre of the rectangle A.
        :param dir_X_A: a vector with length and direction corresponding to the X side of the rectangle A.
        """
        # TODO make unit tests, vizualize
        screens = [dir_X_A, dir_Y_A, dir_X_B, dir_Y_B]  # vector on which the projections are performed (these don't need to be normalized).

        corners_A = [centre_A + dir_X_A/2 + dir_Y_A/2, centre_A - dir_X_A/2 - dir_Y_A/2,
                     centre_A + dir_X_A/2 - dir_Y_A/2, centre_A - dir_X_A/2 + dir_Y_A/2]

        corners_B = [centre_B + dir_X_B/2 + dir_Y_B/2, centre_B - dir_X_B/2 - dir_Y_B/2,
                     centre_B + dir_X_B/2 - dir_Y_B/2, centre_B - dir_X_B/2 + dir_Y_B/2]

        for screen in screens:
            proj_A = []
            proj_B = []
            for c in range(4):
                proj_A.append(np.dot(screen, corners_A[c]))
                proj_B.append(np.dot(screen, corners_B[c]))
            if not (max(proj_A) >= max(proj_B) >= min(proj_A) or
                    max(proj_A) >= min(proj_B) >= min(proj_A) or
                    max(proj_B) >= min(proj_A) >= min(proj_B)):
                return False
                # if not(max(proj_A) > max(proj_B) > min(proj_A))
        return True

    def resolve_grabbability(self, layer_idx: int) -> Set[str]:
        """
        Recursively determines whether a cube is in a set of cubes that all block each other
        # = cube:

        ## - can be grabbed, ## - cannot be grabbed ### - cannot be grabbed   ### - can be grabbed
        #                    ##                     ###                       # #
                                                    ###                       ###
        :return: a Set of cube ids.
        """
        layer = self.layers[layer_idx]
        grabbable_cubes = set()  # cube can be grabbed right away
        ungrabbable_cubes = set()  # cube cannot be grabbed right now but may be in the future
        for cube_idx in layer:
            cube: Cube = layer[cube_idx]
            cube.dfs(grabbable_cubes, ungrabbable_cubes)
        # print(f"grabbable_cubes:{grabbable_cubes}")
        # print(f"ungrabbable_cubes:{ungrabbable_cubes}")
        return ungrabbable_cubes

    def get_grabbable_cubes(self) -> List[Cube]:
        """
        :return: a list of cubes that can be grabbed sorted by layer. (highest layer on first positions).
         Does not take into account if a cube is obstructed by another cube on top of it.
        """
        ret = []
        for layer in reversed(self.layers):
            for key in layer:
                cube:Cube = layer[key]
                if cube.grabable():
                    ret.append(cube)
        return ret


def main():
    layers = Layers()
    T_bo = np.array([[ 0.67119855, -0.70919687, -0.21571362,  0.        ],
       [ 0.7374351 ,  0.60923046,  0.29159504,  0.        ],
       [-0.07537898, -0.35479295,  0.93190128,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    l = layers.__get_cube_layer(T_bo)
    T_to = layers.__align_cube_with_layer(T_bo, l)
    u = layers.rectangles_colide(np.array((0, 0)), np.array((1, 0)), np.array((0, 1)),
                             np.array((0.0, 0.0)), np.array((1, 1)), np.array((1, -1)))
    print(u)
if __name__ == "__main__":
    main()