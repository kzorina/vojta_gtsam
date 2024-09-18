
import numpy as np



class Buffer:
    def __init__(self, min_size = 512, max_size = 1024):
        self.time_stamps = []
        self.poses = []
        self.min_size = min_size
        self.max_size = max_size

    def append(self, time_stamp: float, pose):
        """
        Append element to the buffer and delete old chunk if necessary.
        """
        self.time_stamps.append(time_stamp)
        self.poses.append(pose)
        if len(self.time_stamps) > self.max_size:
            idx_delete = self.max_size-self.min_size
            del self.time_stamps[0:idx_delete]
            del self.poses[0:idx_delete]

    def querry(self, ts):
        """
        Return closest element of the buffer.

        If querried time stamp is outside of the buffer boundaries,
        return None values.

        Input: querried timestamp

        Outputs:
            ts_querry: closest timestamp in buffer
            pose: corresponding pose
        """
        if len(self.time_stamps) == 0:
            return None, None
        if (ts > self.time_stamps[-1] or ts < self.time_stamps[0]):
            return None, None
        idx = np.searchsorted(self.time_stamps, ts)
        pose = self.poses[idx]
        ts_querry = self.time_stamps[idx]
        return ts_querry, pose


if __name__ == '__main__':
    pb = Buffer(min_size=3, max_size=7)
    t = 0
    T = np.eye(4)

    query_t = 5

    for i in range(15):
        T = T.copy()
        T[0,0] = -t
        pb.append(t, T)
        ts_querry, pose = pb.querry(query_t)
        print(ts_querry, pose)
        t += 1