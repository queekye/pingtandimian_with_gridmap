import matplotlib.pyplot as plt
import numpy as np


class grid_map:
    def __init__(self, sd, map_dim, global_pixel_meter, p):
        # TerrianMap 构建高度图的对象
        np.random.seed(sd)
        self.map_dim = map_dim
        self.global_pixel_meter = global_pixel_meter
        side = map_dim * global_pixel_meter
        self.side = side
        self.map_matrix = np.asarray(np.random.random([map_dim, map_dim]) < p, dtype=np.float32)

    def get_local_index(self, loc):
        side = self.side
        gpm = self.global_pixel_meter
        index = np.asarray((loc + side / 2) / gpm, dtype=np.int32)
        return index

    def false_loc(self, loc):
        index = self.get_local_index(loc)
        bool = False
        if (index >= 0).all and (index < self.map_dim).all():
            bool = self.map_matrix[index[0], index[1]] == 1
        return bool

    def plot(self):
        plt.imshow(-self.map_matrix, cmap='bone', origin='lower')
        plt.xticks(())
        plt.yticks(())
        plt.show()


if __name__ == '__main__':
    sed = np.random.randint(1, 10000)
    m = grid_map(sed, 32, 10, 0.3)
    m.plot()

