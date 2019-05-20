import cv2
import numpy as np
import os
from timeit import default_timer as timer
from PIL import ImageTk, Image
from tkinter import *
from tkinter import filedialog
import pickle


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


def task1(img,descriptor1,kp1, img1):

    # cv2.imwrite("task1_sift1.jpg", task1_sift1)
    # cv2.imwrite("task1_sift2.jpg", task1_sift2)
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    task1_sift1, descriptor2, kp2 = draw_key(img_gray)
    good = find_knn_pair(descriptor1, descriptor2)

    # match_img_knn = cv2.drawMatchesKnn(img, kp1, img1, kp2, [[m] for m in good], None, flags=2)
    # cv2.imwrite('task1_matches_knn.jpg', match_img_knn)
    # print(good)
    if len(good) < 10:
        return False
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()
    print(len(mask[mask == 1]))
    #
    if len(mask[mask == 1]) > 10:
        h, w,c = img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)
        img1 = cv2.polylines(img1, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        cv2.imshow("match image", img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True


def compareAll(sample,dir,x1,y1,width,height):
    img1 = fileRead(sample)
    x1 = int(x1)
    y1 = int(y1)
    width = int(width)
    height = int(height)
    img1 = img1[x1:x1+width,y1:y1+height]
    cv2.imshow("query image", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    task1_sift1, descriptor1, kp1 = draw_key(img_gray)
    query_res = []
    res_path = "res_data/"

    for root, subdirs, files in os.walk(dir):

        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img2 = fileRead(os.path.join(root, filename))

                b = task1(img1,descriptor1,kp1, img2)
                if b:
                    query_res.append(filename)
                    cv2.imwrite(res_path+"res_"+filename,img2)

    return query_res


def fileRead(path):
    # path1 = "mountain1.jpg"
    # path2 = "mountain2.jpg"
    img1 = cv2.imread(path)
    return img1

def buildPickle(file):
    img_arr = []
    kp_arr = []
    count = 0
    for root, subdirs, files in os.walk(dir):
        print(root)
        for filename in files:
            print(filename)
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img1 = fileRead(os.path.join(root, filename))
                img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                task1_sift1, descriptor1, kp1 = draw_key(img_gray)
                print(kp1)
                temp = (kp1.pt, kp1.size, kp1.angle, kp1.response, kp1.octave,
                        kp1.class_id, descriptor1)
                img_arr.append([img1,temp])
        p_file = open(file+"img"+str(count)+".pkl","wb")
        pickle.dump(img_arr,p_file)
        p_file.close()
        count+=1
        print(root+" complete")

def loadPickle(file):
    p_file = open(file,"rb")
    img_arr = pickle.load(p_file)
    p_file.close()
    return img_arr

def openfile(root):
    filename = filedialog.askopenfilename(initialdir=dir, title="Select File",
                                          filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    img = PhotoImage(file="paris_1/paris/eiffel/paris_eiffel_000000.jpg")
    panel = Label(root,image=img).grid(row=5)

if __name__ == '__main__':
    dir = "paris_1/paris/eiffel"
    path1 = "paris_1/paris/general/paris_general_002985.jpg"
    x1,y1 = 175.000000,7.000000
    width,height = 590.000000,907.000000

    start = timer()
    result = compareAll(path1,dir,x1,y1,width,height)
    end = timer()
    print("time elapse: {}", end - start)
    with open("eiffel.txt","w") as f:

        for res in result:
            f.write(res+"\n")

    # buildPickle("pickle")

            # current_img = ""
            # root = Tk()
            #
            # Label(root, text = "x1,y1 :").grid(row=0)
            # Label(root, text="x2,y2 :").grid(row=1)
            # Label(root, text="x3,y3 :").grid(row=2)
            # Label(root, text="x4,y4 :").grid(row=3)
            #
            #
            # e1 = Entry(root)
            # e2 = Entry(root)
            # e3 = Entry(root)
            # e4 = Entry(root)
            #
            # e1.grid(row=0, column=1)
            # e2.grid(row=1, column=1)
            # e3.grid(row=2, column=1)
            # e4.grid(row=3, column=1)
            #
            # root.bind('<Return>',openfile)
            # Button(root, text='Quit', command=root.quit).grid(row=3, column=0, sticky=W, pady=4)
            #
            # b = Button(root, text='Input Image', command=lambda:openfile(root)).grid(row=3, column=1, sticky=W, pady=4)
            #
            #
            # mainloop()