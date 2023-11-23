import numpy as np
from Grip import Grip
from typing import List, Dict, Set, Tuple
class Cube():
    def __init__(self, idx:str, T_to:np.ndarray, cube_side = 0.05):
        self.idx: str = idx
        self.T_to: np.ndarray = T_to
        self.cube_side: float = cube_side
        self.grips: List[Grip] = [Grip(T_to, rot=0, offset=0, cube_side=self.cube_side, colision_width=0.05),  # sorted by preference
                                  Grip(T_to, rot=np.pi / 2, offset=0, cube_side=self.cube_side, colision_width=0.05),
                                  Grip(T_to, rot=0, offset=0.02, cube_side=self.cube_side, colision_width=0.05),
                                  Grip(T_to, rot=0, offset=-0.02, cube_side=self.cube_side, colision_width=0.05),
                                  Grip(T_to, rot=np.pi/2, offset=0.02, cube_side=self.cube_side, colision_width=0.05),
                                  Grip(T_to, rot=np.pi/2, offset=-0.02, cube_side=self.cube_side, colision_width=0.05)]

    def get_valid_grips(self):
        ret = []
        for grip in self.grips:
            if len(grip.collisions) == 0:
                ret.append(grip)
        return ret

    def get_collision_rectangle(self) -> Tuple[np.ndarray]:
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
        ret = (self.T_to[:2, 3],
               self.T_to[0:2, 0]*self.cube_side,
               self.T_to[0:2, 1]*self.cube_side)
        return ret

    def grabable(self) -> bool:
        for grip in self.grips:
            if len(grip.collisions) == 0:
                return True
        return False

    def dfs(self, grabbable_cubes: Set, ungrabbable_cubes: Set) -> int:
        """
        a recursive function that is used to find
        """
        if self.idx in grabbable_cubes:
            return 1
        if self.idx in ungrabbable_cubes:
            return 0
        if self.grabable():
            grabbable_cubes.add(self.idx)
            return 1
        else:
            ungrabbable_cubes.add(self.idx)
            for grip in self.grips:
                resolved_collisions = 0
                for collision in grip.collisions:
                    resolved_collisions += collision.dfs(grabbable_cubes, ungrabbable_cubes)
                if resolved_collisions == len(grip.collisions):
                    ungrabbable_cubes.remove(self.idx)
                    grabbable_cubes.add(self.idx)
                    return 1
            return 0

def main():
    pass

if __name__ == "__main__":
    main()