import numpy as np
from scipy.spatial import cKDTree

import pdb

# [num_cam, batch_size, num_query, num_points, 2(x, y)]
point= np.random.rand(6, 1, 9354, 4, 2)
point = point.reshape(6*1*9354*4, -1)


unique_coords, unique_indices = np.unique(point, axis=0, return_index=True)

# 构建 KD 树
tree = cKDTree(unique_coords)

distances, neighbor_indices = tree.query(point, k=4+1)

pdb.set_trace()

neighbor_indices = unique_indices[neighbor_indices]

points = unique_coords[neighbor_indices]

