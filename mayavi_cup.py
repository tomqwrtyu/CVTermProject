import numpy as np
from mayavi import mlab
from tvtk.api import tvtk  # 注意你需要這樣導入tvtk

# Define the parameters for the 'cup'
n = 100
radius_bottom = 1
radius_top = 2
height = 5

# Generate the meshgrid for the sides of the cup
theta = np.linspace(0, 2.*np.pi, n)
r = np.linspace(radius_bottom, radius_top, n)
Theta, R = np.meshgrid(theta, r)

# Convert to cartesian coordinates
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
Z = np.linspace(0, height, n).reshape(-1, 1)

Z = np.tile(Z, (1, n))  # This makes Z a 2D array of the same shape as X and Y



# Plot
mesh = mlab.mesh(X, Y, Z, color=(0.5, 0.5, 0.5))  # grey color

# azimuth是在x-y平面上面的角度,也可以說是物體繞著z軸旋轉, azimuth=0是從x軸正方向看過去, azimuth=90是從y軸正方向看過去
# elevation是從x-y平面往z軸方向看的角度, elevation=0是從Z軸正上方看過去, elevation=90是平行著x-y平面正前方看過去, elevation=180是從Z軸正下方看過去
# 因為繞著杯狀物體一整圈看到的畫面是相同的，所以在實驗當中我們固定azimuth，藉由調整elevation以及roll來觀察物體的不同角度
# roll 轉動的是相機本身，而不是物體
'''
使用 azimuth = 0, elevation = 90, roll = 270看到的cup應該為:
-----------------
\              /
 \            /
  \          /
   \        /
    \      /
     \    /
      \ _/
'''

mlab.view(elevation=90, distance=20, focalpoint=(0, 0, 1), roll=270)


mlab.show()  # This will open a window with your plot


