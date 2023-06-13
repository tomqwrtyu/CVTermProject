from vpython import *
import numpy as np

def stl_to_triangles(fileinfo): 
    fd = open(fileinfo, mode='rb')
    tris = [] # list of triangles to compound
    fd.seek(0)
    fList = fd.readlines()

    # Decompose list into vertex positions and normals
    vs = []
    for line in fList:
        FileLine = line.split( )
        if FileLine[0] == b'facet':
            N = vec(float(FileLine[2]), float(FileLine[3]), float(FileLine[4]))
        elif FileLine[0] == b'vertex':
            vs.append( vertex(pos=vec(float(FileLine[1]), float(FileLine[2]), float(FileLine[3])), normal=N, color=color.white) )
            if len(vs) == 3:
                tris.append(triangle(vs=vs))
                vs = []
                    
    return compound(tris)

if __name__ == '__main__':
    scene = canvas(title = "CV term project", width = 1280, height = 720, x = 0, y = 0,\
                   background = vector(0.14, 0.24, 0.38))
    
    h = 1
    L = 1000
    v = 0.03
    t = 0
    dt = 0.01
    scene.camera.pos = vec(0, 400, 820)
    scene.camera.axis = vec(0, -160, -270)
    a_floor = box(pos = vector(0, 0, 0), length = L, height = h, width = L, color = vector(1.2 * 186 / 255, 1.2 * 153 / 255, 80 / 255))

    man = stl_to_triangles("model/cupper.stl")
    many = man.size.y
    man.pos = vec(-200, many // 2, 0)
    man.color = color.white
    # part = stl_to_triangles('Part1.stl')
    # part.pos = vec(-200,0,0)
    # part.color = color.orange
    cam = stl_to_triangles("model/Intel_RealSense_Depth_Camera_D435.stl")
    cam.size *= 10
    cam.pos = vec(0, cam.size.y + 112, 500)

    i = 0
    while True:
        sleep(1 / 1000)
        man.pos += dt * 50 * vec(np.sin((i % 360) / 180 * np.pi), 0, 0)
        i += 2
