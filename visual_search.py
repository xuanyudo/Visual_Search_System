import cv2
import numpy as np

def draw_key(img):
    sift1 = cv2.xfeatures2d.SIFT_create()
    kp = sift1.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp, des = sift1.compute(img, kp)
    # cv2.imshow("key poiuuint", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img2, des, kp


def find_knn_pair(descriptor1, descriptor2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    # print(matches)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return good


def select10(matchesMask):
    mask = np.asarray(matchesMask)
    mask1 = np.zeros(len(mask))
    inlier = np.argwhere(mask == 1)
    # start = np.random.randint(0, len(inlier) - 10)
    inlier = inlier.transpose()

    # inlier = inlier[0][start:start + 10]

    for dot in inlier:
        mask1[dot] = 1
    return mask1

def task1(img, img1):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    task1_sift1, descriptor1, kp1 = draw_key(img_gray)
    task1_sift2, descriptor2, kp2 = draw_key(img_gray1)

    cv2.imwrite("task1_sift1.jpg", task1_sift1)
    cv2.imwrite("task1_sift2.jpg", task1_sift2)

    good = find_knn_pair(descriptor1, descriptor2)

    match_img_knn = cv2.drawMatchesKnn(img, kp1, img1, kp2, [[m] for m in good], None, flags=2)
    cv2.imwrite('task1_matches_knn.jpg', match_img_knn)
    # print(good)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(M)
    matchesMask = mask.ravel().tolist()
    print(len(matchesMask))
    #
    h, w = img_gray.shape
    w1, h1 = img_gray1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    padding_left = 0
    padding_top = 0
    padding_down = 0
    padding_right = 0

    dst = cv2.perspectiveTransform(pts, M)
    img1 = cv2.polylines(img1, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    if min(dst[:][:, :, 0]) < 0:
        padding_left = int(abs(min(dst[:][:, :, 0])))
    if max(dst[:][:, :, 0]) > w1:
        padding_right = int(max(dst[:][:, :, 0]))
    if min(dst[:][:, :, 1]) < 0:
        padding_top = int(abs(min(dst[:][:, :, 1])))
    if max(dst[:][:, :, 1]) > h1:
        padding_down = int(max(dst[:][:, :, 1]))
    outputImage = cv2.copyMakeBorder(
        img1,
        padding_top,
        padding_down,
        padding_left,
        padding_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    dst_new = []
    for d in dst[:][:]:
        dst_new.append([padding_left + int(d[0][0]), padding_top + int(d[0][1])])
    dst_new = np.float32([dst_new]).reshape(-1, 1, 2)

    # ``

    matrix = cv2.getPerspectiveTransform(pts, dst_new)
    result = cv2.warpPerspective(img, matrix, (outputImage.shape[1], outputImage.shape[0]))
    result[padding_top:padding_top + h, padding_left:padding_left + w] = outputImage[padding_top:padding_top + h,
                                                                         padding_left:padding_left + w]

    cv2.imwrite("task1_pano.jpg", result)
    mask1 = select10(matchesMask)
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=mask1,  # draw only inliers
                       flags=2)

    match_10 = cv2.drawMatches(img, kp1, img1, kp2, good, None, **draw_params)

    cv2.imwrite("task1_matches.jpg", match_10)

if __name__ == '__main__':
    path1 = "paris_1/paris/eiffel/paris_eiffel_000005.jpg"
    path2 = "paris_1/paris/eiffel/paris_eiffel_000014.jpg"
    # path1 = "mountain1.jpg"
    # path2 = "mountain2.jpg"

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    cv2.imshow("img1",img1)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    task1(img1,img2)