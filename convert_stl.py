import cv2
import pyrealsense2 as rs
from vpython import *
import numpy as np
import os
import glob
import pickle

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

def initCamera():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.depth_units):
        depth_sensor.set_option(rs.option.depth_units,0.001)
    depth_scale = depth_sensor.get_depth_scale()
    print("Camera using depth scale: {}".format(depth_scale))
    align = rs.align(rs.stream.color)

    return pipeline, align, depth_scale

if __name__ == '__main__':
    # initializa camera
    # pipeline, align, depth_scale = initCamera()

    # initialize SIFT detector, load feature keypoints & descriptions
    sift_detector = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    Lowes_ratio = 0.6
    raw_features_path = "./KEBUKE_feature"

    features = []
    for f in glob.glob(os.path.join(raw_features_path, "*.pickle")):
        features.append(pickle.load(f)) # this should maintain a list with [kpts:list, des:list, relative point to cup origin:list]

    # initialize feature matcher
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # load model

    # building scene
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
    man.color = vector(0.95, 0.95, 0.95)
    cam = stl_to_triangles("model/Intel_RealSense_Depth_Camera_D435.stl")
    cam.size *= 10
    cam.pos = vec(0, cam.size.y + 112, 500)

    i = 0
    while True:
        rate(120)
        # get frame
        # frames = pipeline.wait_for_frames()
        # if not frames:
        #    continue
        # aligned_frames = align.process(frames)   
        # depth_frame = aligned_frames.get_depth_frame() 
        # color_frame = aligned_frames.get_color_frame()
        # aligned_intrinsic = aligned_frames.profile.as_video_stream_profile().intrinsics

        # new_frame = np.asanyarray(color_frame.get_data())

        # detect bbox
        # bboxes = []


        # detect features
        # bbfeatures = []
        # for bbox in bboxes:
        #     x1, y1 = bbox[0]
        #     x2, y2 = bbox[1]
        #     bbfeatures.append(sift_detector.detectAndCompute(cv2.cvtColor(new_frame[x1:x2, y1:y2], cv2.COLOR_BGR2GRAY), None))

        # kdtree feature matching
        # cup_world_axes = []
        # for bbf in bbfeatures:
        #     bbktps, bbdes = bbf

        #     for fkpts, fdes, frp in features:
        #         matches = flann.knnMatch(bbdes, fdes, k=2)
        #         # Need only good matches, so create a mask
        #         matchesMask = np.argwhere((lambda match: [tup[0].distance < Lowes_ratio * tup[1].distance for tup in match])(matches))
        #         # select 1 good point (RANSAC)
        #         # goodidx = 

        #         # using that good point to retrieve point relative to cup axis
        #         # relative_cup_point = frp[goodidx]

        #         # transform to world axis
        #         cup_world_axes.append(rs.rs2_deproject_pixel_to_point(aligned_intrinsic, relative_cup_point, depth_scale))

        # update cup position
        man.pos += dt * 120 * vec(np.sin((i % 360) / 180 * np.pi), 0, 0)
        i += 4
