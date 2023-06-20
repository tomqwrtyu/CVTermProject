import cv2
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import os
import glob
import pickle
from ultralytics import YOLO

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

def RANSAC_randPt(fktps, bbktps, matches, err_dist:float = 1, err_pts = 4, k_points = 4, max_iter = 100):
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
        X1 = None
        X2_Prime = None 
        h_inv = np.linalg.pinv(h)
        for idx1, idx2 in match_pts:
            p1 = bbktps[idx1].pt
            p2 = fktps[idx2]

            if isinstance(X1, np.ndarray):
                X1 = np.vstack((X1, np.array([p1[0], p1[1]])))
                X2_Prime = np.vstack((X2_Prime, h_inv @ np.array([p2[0], p2[1], 1]).T))
            else:
                X1 = np.array([p1[0], p1[1]])
                X2_Prime = h_inv @ np.array([p2[0], p2[1], 1]).T
            
        not_match_npts = 0
        
        for i in range(X2_Prime.shape[0]):
            if X2_Prime[i, 2] < 1e-3:
                not_match_npts += 1
                continue
            if np.linalg.norm((X2_Prime[i, :2] / X2_Prime[i, 2]) - X1[i]) >= err_dist:
                not_match_npts += 1

        if not_match_npts <= err_pts:
            return rand_matches[0][0], rand_matches[0][1]
        
        h_set.append([rand_matches[0], not_match_npts])
    
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_shape = (720, 1280)
    out = cv2.VideoWriter('result.mp4', fourcc, 30.0, video_shape)

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
    model = YOLO("best.pt")
    estimated_coordinates_record = []

    while True:
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
        bboxes = []
        results = model(new_frame, verbose=False)

        for i, (result) in enumerate(results):
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                t = int(xyxy[0].item())
                l = int(xyxy[1].item())
                b = int(xyxy[2].item())
                r = int(xyxy[3].item())
                bboxes.append([[l, t], [r, b]])

        cup_world_points = []
        if len(bboxes) > 0:
            # detect features and draw boxes
            bbfeatures = []
            for bbox in bboxes:
                x1, y1 = bbox[0]
                x2, y2 = bbox[1]
                bbfeatures.append([bbox, sift_detector.detectAndCompute(cv2.cvtColor(new_frame[x1:x2, y1:y2], cv2.COLOR_BGR2GRAY), None)])

            for bbox in bboxes:
                x1, y1 = bbox[0]
                x2, y2 = bbox[1]
                cv2.rectangle(new_frame, (y1, x1), (y2, x2), color=(0, 255, 0), thickness=2)

            # kdtree feature matching
            for bbox, (bbktps, bbdes) in bbfeatures:
                for fkpts, fdes, frpts in features:
                    xo, yo = min(bbox[0][0], bbox[1][0]), min(bbox[0][1], bbox[1][1])
                    matches = np.array(flann.knnMatch(bbdes, fdes, k=2))
                    # Need only good matches, so create a mask
                    matchesMask = np.argwhere((lambda match: [tup[0].distance < Lowes_ratio * tup[1].distance for tup in match])(matches)).flatten()
                    # select 1 good point (RANSAC)
                    if len(matchesMask) < 4:
                        continue
                    goodbbidx, goodfidx = RANSAC_randPt(fkpts, bbktps, matches[matchesMask], err_pts=2, max_iter=100)
                    
                    # using that good point to retrieve point relative to cup axis
                    relative_cup_point = frpts[goodfidx]

                    # transform to world axis
                    by, bx = bbktps[goodbbidx].pt
                    x, y = int(xo + bx), int(yo + by)
                    cv2.circle(new_frame, (y, x), radius=2, color=(0, 0, 255), thickness=-1)
                    depth = depth_frame.get_distance(y, x)
                    if depth == camera_extrinsics.translation:
                        continue
                    camera_axis_point = rs.rs2_deproject_pixel_to_point(aligned_intrinsic, (y, x), depth)
                    world_axis_point = rs.rs2_transform_point_to_point(camera_extrinsics, camera_axis_point)
                    estimated_coordinates_record.append([world_axis_point[0], world_axis_point[1], depth])
                    cv2.putText(new_frame, "Estimated world coordinates: ({:2f}, {:2f}, {:2f}) (m).".format(world_axis_point[0], world_axis_point[1], depth),\
                        org=(20, 60), fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=2, fontScale=1, color=(0, 0, 0))

        # cv2.putText(new_frame, "Number of boxes detected: {}.".format(len(bboxes)),\
        #                 org=(20, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=2, fontScale=1, color=(0, 0, 0))
        out.write(new_frame)
        cv2.imwrite("precision test.jpg", new_frame)
        cv2.imshow("Realsense RGB", new_frame)
        key = cv2.waitKey(1)
        if key == 27:
            np.save("coordinates", np.array(estimated_coordinates_record))
            cv2.destroyAllWindows()
            break