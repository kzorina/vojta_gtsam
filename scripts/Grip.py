import numpy as np

class Grip():
    def __init__(self, T_to:np.ndarray, rot:float = 0, offset:float = 0, cube_side = 0.05, colision_width = 0.05, colision_length = 0.07):
        # transformation from the grip to the cube frame of refference
        self.z_offset = 0.15
        self.T_og = np.array(((np.cos(rot), -np.sin(rot), 0, 0),
                              (np.sin(rot), np.cos(rot), 0, 0),
                              (0, 0, 1, self.z_offset),
                              (0, 0, 0, 1)))
        T_to_shifted = np.copy(T_to)
        T_to_shifted[:2, 3] += T_to_shifted[:2, :2]@self.T_og[:2, :2]@np.array((offset, 0))
        self.T_tg = T_to_shifted @ self.T_og  # grip to table
        self.collisions = []
        self.cube_side = cube_side
        self.collision_width = colision_width  # meters
        self.collision_length = colision_length

    def get_collision_rectangle(self):
        """
        (rectangle diagram:)
        Y    c ------- d
        ^      |     |
        |      |  o<-|---centre
        |      |     |
        |    a ------- b
        |_________________> X
        :return: (np.array([centre_x, centre_y]), (a - b), (d - b))
        """
        # centre, dir_X, dir_Y
        ret = (self.T_tg[0:2, 3],
               self.T_tg[0:2, 0]*self.collision_width,
               self.T_tg[0:2, 1]*(2*self.collision_length + self.cube_side))
        return ret


def main():
    pass

if __name__ == "__main__":
    main()