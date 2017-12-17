import sys
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import matplotlib.cm as cm
import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# Used to check normal vectors with to determine overhang angle
GRAV_CONST = [0,0,-1]
GRAV_CONST_UP = [0,0,1]

# 3 args: stl file, gravity vector, max overhang angle in radians
# optimize by subtracting theta from max_overhang
max_overhang = np.pi/2 - float(sys.argv[3])
new_grav = sys.argv[2].split(',')
for index, item in enumerate(new_grav):
    new_grav[index] = float(item)
stl_file = sys.argv[1]

overhang_idx = []
overhang_vectors = []

# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)
axes.set_xlabel('x axis')
axes.set_ylabel('y axis')
axes.set_zlabel('z axis')

# Load the STL files and rotate based on given gravity vector and recalculate normals
your_mesh = mesh.Mesh.from_file(stl_file)
your_mesh.rotate(np.cross(GRAV_CONST, new_grav), angle_between(new_grav, GRAV_CONST))
non_overhang_vectors = your_mesh.vectors.tolist()
your_mesh.update_normals()

# Determine if the tesselation exceeds the max overhang
your_normals = your_mesh.normals
for index, x in enumerate(your_normals):
    angle = min(angle_between(x, GRAV_CONST), angle_between(x, GRAV_CONST_UP))
    if angle < max_overhang:
        overhang_idx.append(index)

# Add angles that have overhang to new list, delete them from the original list
for i in reversed(overhang_idx):
    overhang_vectors.append(your_mesh.vectors[i])
    non_overhang_vectors.pop(i)

# Convert to np arrays
overhang_vectors_np = np.array(overhang_vectors)
non_overhang_vectors_np = np.array(non_overhang_vectors)

# Add non-overhang vectors and overhang vectors to the plot 
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(non_overhang_vectors_np, facecolors='b'))
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(overhang_vectors_np, facecolors='r'))

# Auto scale to the mesh size
scale = your_mesh.points.flatten(-1)
axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
pyplot.show()