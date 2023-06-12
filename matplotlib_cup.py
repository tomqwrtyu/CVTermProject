import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the parameters for the 'cup'
n = 40
radius_bottom = 1
radius_top = 0.6
height = 3

maxX = height
maxY = height
maxZ = height
dx = 0
dy = 0
dz = 0

# Generate the meshgrid for the sides of the cup
theta = np.linspace(0, 2.*np.pi, n)
r = np.linspace(radius_bottom, radius_top, n)
Theta, R = np.meshgrid(theta, r)

# Convert to cartesian coordinates
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
Z = np.linspace(0, height, n).reshape(-1, 1)

# Generate the meshgrid for the bottom of the cup
theta2 = np.linspace(0, 2.*np.pi, n)
r2 = np.linspace(0, radius_bottom, n)
Theta2, R2 = np.meshgrid(theta2, r2)

# Convert to cartesian coordinates
X2 = R2 * np.cos(Theta2)
Y2 = R2 * np.sin(Theta2)
Z2 = np.zeros_like(R2)

# Create the 3D plot    
fig = plt.figure(figsize=(10,10))  # Increase figure size to 10x10 inches
ax1 = fig.add_subplot(projection='3d')
ax1.set_zlim(0,maxZ)
# ax1.set_ylim(0,maxY)
# ax1.set_xlim(0,maxX)

# Plot the sides of the cup
ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, color='k', edgecolors='w')
ax1.set_xticks(np.linspace(0, height, maxX))
ax1.set_yticks(np.linspace(0, height, maxY))
# Plot the bottom of the cup
# ax1.plot_surface(X2, Y2, Z2, rstride=5, cstride=5, color='k', edgecolors='w')

# plt.waitforbuttonpress()
plt.show()



# def rotation_matrix(roll, pitch, yaw):
#     """Generate a rotation matrix given roll, pitch, and yaw angles."""
#     R_x = np.array([[1, 0, 0],
#                     [0, np.cos(roll), -np.sin(roll)],
#                     [0, np.sin(roll), np.cos(roll)]])
    
#     R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
#                     [0, 1, 0],
#                     [-np.sin(pitch), 0, np.cos(pitch)]])
    
#     R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
#                     [np.sin(yaw), np.cos(yaw), 0],
#                     [0, 0, 1]])
    
#     R = np.dot(R_z, np.dot(R_y, R_x))
#     return R

# # Define the roll, pitch, and yaw angles
# roll = np.radians(90)  # x rotation
# pitch = np.radians(30)   # y rotation
# yaw = np.radians(0)    # z rotation

# # Generate the rotation matrix
# R = rotation_matrix(roll, pitch, yaw)


# for i in range(n):
#     height = Z[i,0]
#     v = np.array([X[i,0], Y[i,0], height])
#     for j in range(n):
#         v = np.array([X[i,j], Y[i,j], height])
#         v = np.dot(R, v)
#         X[i,j] = v[0]
#         Y[i,j] = v[1]
#     Z[i,0] = v[2]


# for i in range(n):
#     # height = Z2[i]
#     height = 0.0
#     v = np.array([X2[i,0], Y2[i,0], height])
#     for j in range(n):
#         v = np.array([X2[i,j], Y2[i,j], height])
#         v = np.dot(R, v)
#         X2[i,j] = v[0]
#         Y2[i,j] = v[1]
#     Z2[i,0] = v[2]

# # Create the 3D plot
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.set_zlim(-1*max(Z),max(Z))

# # Plot the sides of the cup
# ax2.plot_surface(X, Y, Z, rstride=5, cstride=5, color='k', edgecolors='w')

# # Plot the bottom of the cup
# # ax2.plot_surface(X2, Y2, Z2, rstride=5, cstride=5, color='k', edgecolors='w')

# # plt.waitforbuttonpress()
# plt.show()