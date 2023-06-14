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

    return pipeline, align

def RANSAC_randPt(fktps, bbktps, matches, err_dist = 1, err_pts = 4, k_points = 4, max_iter = 100):
    match_pts = np.array([[match[0].queryIdx, match[0].trainIdx] for match in matches])
    h_set = []
    for _ in range(max_iter):
        rand_matches = [match_pts[x] for x in np.random.choice(len(match_pts), k_points, replace = False)]
        #find homography
        A = np.zeros((2 * k_points, 9))

        for i, (idx1, idx2) in enumerate(rand_matches):
            p1 = bbktps[idx1].pt
            p2 = fktps[idx2]
            A[2 * i] = [p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1], -p2[0]]
            A[2 * i + 1] = [0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1], -p2[1]]

        _, _, vt = np.linalg.svd(A)
        h = vt[-1].reshape(3, 3)
        h = (1 / h.item(8)) * h

        #test homography
        X1_Prime = None
        X2 = None 
        for idx1, idx2 in match_pts:
            p1 = bbktps[idx1].pt
            p2 = fktps[idx2]

            if isinstance(X1_Prime, np.ndarray):
                X1_Prime = np.vstack((X1_Prime, h @ np.array([p1[0], p1[1], 1]).T))
                X2 = np.vstack((X2, np.array([p2[0], p2[1]])))
            else:
                X1_Prime = h @ np.array([p1[0], p1[1], 1]).T
                X2 = np.array([p2[0], p2[1]])
            
        match_npts = 0
        
        for i in range(X1_Prime.shape[0]):
            if X1_Prime[i, 2] < 1e-3:
                continue
            if np.linalg.norm((X1_Prime[i, :2] / X1_Prime[i, 2]) - X2[i]) < err_dist:
                match_npts += 1

        if match_npts < err_pts:
            return rand_matches[0][0], rand_matches[0][1]
        
        h_set.append([rand_matches[0], match_npts])
    
    h_set.sort(key=lambda x:x[1])
    return h_set[0][0][0], h_set[0][0][1]

if __name__ == '__main__':
    # initializa camera
    # world axis origin locate at the bottom ground of the camera, the depth camera is located (-30, 125, 500)
    # camera: x: → y: ↓ z: o
    # world: x: → y: ↑ z: x (using the cooridinates of vpython)
    # need to rotate x: 180 deg, y: 0 deg, z: 0 deg, translate (-30, 125, 500) (mm)
    pipeline, align = initCamera()
    camera_extrinsics = rs.extrinsics()
    camera_extrinsics.rotation = [float(x) for x in [1, 0, 0, 0, np.cos(np.pi), -np.sin(np.pi), 0, np.sin(np.pi), np.cos(np.pi)]]
    camera_extrinsics.translation = [-0.03, 0.125, 0.5] # location of "depth lens", specified in meter

    # initialize SIFT detector, load feature keypoints & descriptions
    sift_detector = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    Lowes_ratio = 0.6
    raw_features_path = "./KEBUKE_feature"

    features = []
    for ff in glob.glob(os.path.join(raw_features_path, "*.pickle")):
        with open(ff, 'rb') as f:
            features.append(pickle.load(f)) # this should maintain a list with [kpts:list, des:ndarray, relative point to cup origin:list]

    # initialize feature matcher
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # load model

    # building scene
    scene = canvas(title = "CV term project", width = 1900, height = 900, x = 0, y = 0,\
                   background = vector(0.14, 0.24, 0.38))
    
    h = 1
    L = 1000
    v = 0.03
    t = 0
    dt = 0.01
    scene.camera.pos = vec(0, 400, 820)
    scene.camera.axis = vec(0, -160, -270)
    a_floor = box(pos = vector(0, 0, 0), length = L, height = h, width = L, color = vector(1.2 * 186 / 255, 1.2 * 153 / 255, 80 / 255))

    cupper = stl_to_triangles("model/cupper.stl")
    cupper_origin = vec(0, cupper.size.y // 2, -cupper.size.z // 2)
    cupper.pos = vec(0, 0, 0) + cupper_origin
    cupper.color = vector(0.95, 0.95, 0.95)
    cuppers = [cupper]
    num_cuppers_activate = 1
    last_deactivate_idx = -1

    cam = stl_to_triangles("model/Intel_RealSense_Depth_Camera_D435.stl")
    cam.size *= 10
    cam.pos = vec(0, 125, cam.size.z // 2 + 500)

    while True:
        rate(120)
        # get frame
        frames = pipeline.wait_for_frames()
        if not frames:
           continue
        aligned_frames = align.process(frames)   
        depth_frame = aligned_frames.get_depth_frame() 
        color_frame = aligned_frames.get_color_frame()
        aligned_intrinsic = aligned_frames.profile.as_video_stream_profile().intrinsics

        new_frame = np.asanyarray(color_frame.get_data())

        # detect bbox
        bboxes = [[[240, 480], [719, 800]]]

        if len(bboxes) > 0:
            # detect features and draw boxes
            bbfeatures = []
            for bbox in bboxes:
                x1, y1 = bbox[0]
                x2, y2 = bbox[1]
                cv2.rectangle(new_frame, (y1, x1), (y2, x2), color=(0, 255, 0), thickness=2)
                bbfeatures.append([bbox, sift_detector.detectAndCompute(cv2.cvtColor(new_frame[x1:x2, y1:y2], cv2.COLOR_BGR2GRAY), None)])

            cv2.putText(new_frame, "Number of boxes detected: {}.".format(len(bboxes)),\
                        org=(20, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=2, fontScale=0.6, color=(0, 0, 0))
            cv2.imshow("Realsense RGB", new_frame)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break

            # kdtree feature matching
            cup_world_points = []
            for bbox, (bbktps, bbdes) in bbfeatures:
                for fkpts, fdes, frpts in features:
                    xo, yo = min(bbox[0][0], bbox[1][0]), min(bbox[0][1], bbox[1][1])
                    matches = np.array(flann.knnMatch(bbdes, fdes, k=2))
                    # Need only good matches, so create a mask
                    matchesMask = np.argwhere((lambda match: [tup[0].distance < Lowes_ratio * tup[1].distance for tup in match])(matches)).flatten()
                    # select 1 good point (RANSAC)
                    if len(matchesMask) < 4:
                        continue
                    goodbbidx, goodfidx = RANSAC_randPt(fkpts, bbktps, matches[matchesMask])
                    
                    # using that good point to retrieve point relative to cup axis
                    relative_cup_point = frpts[goodfidx]

                    # transform to world axis
                    by, bx = bbktps[goodbbidx].pt
                    x, y = int(xo + bx), int(yo + by)
                    depth = depth_frame.get_distance(y, x)
                    if depth == camera_extrinsics.translation:
                        continue
                    camera_axis_point = rs.rs2_deproject_pixel_to_point(aligned_intrinsic, (y, x), depth)
                    world_axis_point = rs.rs2_transform_point_to_point(camera_extrinsics, camera_axis_point)
                    cup_world_points.append(np.array(world_axis_point) * 1000 - np.array(relative_cup_point))

        # update cup position
        if len(cuppers) < len(cup_world_points): # need to add new cuppers
            for _ in range(len(cup_world_points) - len(cuppers)):
                cuppers.append(cupper.clone())
                num_cuppers_activate += 1

        #this part can run for a large amount of times since no cupper model is deleted
        elif len(cuppers) > len(cup_world_points): # some cuppers are lost, or depth value is not available
            for i in range(len(cuppers) - len(cup_world_points)):
                if cuppers[-i-1].visible == True:
                    cuppers[-i-1].visible = False
                    num_cuppers_activate -= 1
                    last_deactivate_idx = len(cuppers) - i - 1

        elif num_cuppers_activate < len(cup_world_points) : # some cuppers are again detected, activating the invisible.
            temp = last_deactivate_idx
            
            for i in range(len(cup_world_points) - num_cuppers_activate):
                if cuppers[temp + i].visible == False:
                    cuppers[temp + i].visible = True
                    num_cuppers_activate += 1
                    last_deactivate_idx = temp + i + 1
            
        for cup, (x, y, z) in zip(cuppers, cup_world_points):
            cup.pos = vec(x, y, z) + cupper_origin

