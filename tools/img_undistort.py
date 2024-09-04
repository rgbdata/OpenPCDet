import numpy as np
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    K = np.array([
                    [286.7037963867188, 0.0, 413.3463134765625], 
                    [0.0, 286.7817993164062, 397.1785888671875], 
                    [0.0, 0.0, 1.0]
                ])
    D = np.array([-0.01078350003808737, 0.04842806980013847, -0.04542399942874908, 0.008737384341657162])

    img_dist = cv2.imread('data/caic_bev/test.png')
    # img_ud = cv2.undistort(img_dist, intrinsics, distortion)
    # img_ud = cv2.fisheye.undistortImage(img_dist, K, D)
    size = (img_dist.shape[1],img_dist.shape[0])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, size, cv2.CV_16SC2)


    IMG_W, IMG_H = size[0], size[1]
    plt.figure(figsize=((IMG_W)/72.0,(IMG_H)/72.0),dpi=72.0, tight_layout=True)
    plt.axis([0,IMG_W,IMG_H,0])
    # plt.axis('off') 
    plt.imshow(img_dist[..., ::-1]) #在画布上画出原图
    px = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
    py = np.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]) + 140

    v = map1[px, py, :][:, 0]
    u = map1[px, py, :][:, 1]

    print(u)
    print(v)

    plt.scatter([u], [v], c='r', alpha=0.5, s=8)
    plt.savefig('data/caic_bev/draw_test.jpg', bbox_inches='tight')


    # img_ud = cv2.remap(img_dist, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    # cv2.imwrite('data/caic_bev/result.jpg', img_ud)