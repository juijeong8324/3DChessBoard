import numpy as np
import cv2 as cv
import prepare_box as pb

# The given video and calibration data
video_file = 'data/chessboard2.mp4'
K = np.array([[1.20105746e+03, 0.00000000e+00, 6.34435026e+02],
              [0.00000000e+00, 1.20893667e+03, 3.53765890e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeff = np.array([1.99616322e-01, -1.02188362e+00, -
                      6.70423343e-03, -3.28932459e-03, 3.59774606e+00])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + \
    cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Prepare a 3D box for simple AR
lower_list, upper_list = pb.prepare_box(
    board_cellsize, x1=0, x2=10, y1=0, y2=7)

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Prepare 3D points on a chessboard
obj_points = board_cellsize * \
    np.array([[c, r, 0] for r in range(board_pattern[1])
             for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(
        img, board_pattern, board_criteria)
    if success:
        # 3d 공간에서 obj points에 대응하는 이미지 평면상의 위치를 찾기 위함
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the box on the image
        for b_l, b_u in zip(lower_list, upper_list):
            line_lower, _ = cv.projectPoints(
                b_l, rvec, tvec, K, dist_coeff)  # plane에 projection한 2차원 좌표 찾기
            line_upper, _ = cv.projectPoints(b_u, rvec, tvec, K, dist_coeff)
            # cv.fillPoly(img, [np.int32(line_upper)], (0, 255, 0)) 해당 위치에 두면 lower가 됨

            for num in range(4):  # 옆면을 채우기 위함
                line_side = []
                line_side = np.vstack((line_lower[num % 4], line_lower[(
                    num+1) % 4], line_upper[(num+1) % 4], line_upper[num % 4]))  # 순서 바꾸면 꼬임

                cv.fillPoly(img, [np.int32(line_side)], (93, 104, 115))

            cv.fillPoly(img, [np.int32(line_upper)],
                        (0, 0, 0))  # 그려지는 순서 중요

        # Print the camera position
        # Alternative) `scipy.spatial.transform.Rotation`
        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25),
                   cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27:  # ESC
        break

video.release()
cv.destroyAllWindows()
