#!/usr/bin/env python3
import cv2 as cv
import sys
import numpy as np
import math as m
import os

#------------------------------------------------------------------------------
#--Helper Functions
#------------------------------------------------------------------------------


def bk_MOG_init():
    bk_remove = cv.createBackgroundSubtractorMOG2(detectShadows=False)
    bk_remove.setHistory(100)
    #print(bk_remove.getHistory())

    return bk_remove

def bk_MOG_update_v2(bk_remove,input_frame):

    if not hasattr(bk_MOG_update_v2, "calibrate"):
        bk_MOG_update_v2.calibrate = False

    if not hasattr(bk_MOG_update_v2, "calibrate_counter"):
        bk_MOG_update_v2.calibrate_counter = 100

    if bk_MOG_update_v2.calibrate == False:
        bk_mask = bk_remove.apply(input_frame)
        bk_composite_frame = bk_remove.getBackgroundImage()
    else:
        bk_mask = bk_remove.apply(input_frame,0,0)
        bk_composite_frame = bk_remove.getBackgroundImage()


    white_pixel = cv.countNonZero(bk_mask)
    total_size = input_frame.shape[1] * input_frame.shape[0]
    percent = (float(white_pixel) / float(total_size)) * 100

    #print("percent: "+ str(percent) + "  calibrate_state: " + str(bk_MOG_update_v2.calibrate) + "  calibrate_counter: " + str(bk_MOG_update_v2.calibrate_counter))


    if percent > 70:
        bk_MOG_update_v2.calibrate = False
        bk_MOG_update_v2.calibrate_counter = 0
    elif (bk_MOG_update_v2.calibrate_counter < 100 or percent > 0.5) and bk_MOG_update_v2.calibrate == False:
        bk_MOG_update_v2.calibrate = False
        bk_MOG_update_v2.calibrate_counter += 1
    else:
        bk_MOG_update_v2.calibrate = True

    cut_image = cv.bitwise_and(input_frame, input_frame,mask = bk_mask)

    return bk_mask, bk_composite_frame, cut_image


def bk_MOG_update(bk_remove,input_frame):
     bk_mask = bk_remove.apply(input_frame)
     bk_composite_frame = bk_remove.getBackgroundImage()

     return bk_mask, bk_composite_frame

def fastfeaturedetect(input_img):
    fast = cv.FastFeatureDetector_create()
    fast.setThreshold(12)
    key_points = fast.detect(input_img,None)
    output_img = cv.drawKeypoints(input_img, key_points, None, color=(255,0,0))

    fast_pts_xy = np.empty([0,1,2], dtype=np.float32)
    for item in key_points:
        fast_pts_xy = np.append(fast_pts_xy,[[[np.float32(item.pt[1]),np.float32(item.pt[0])]]],axis=0)

    #print(fast_pts_xy.shape)
    return output_img, fast_pts_xy

def realtime_frame_mapper(input_frame, bk_composite_frame, fast_pts_xy):
    if fast_pts_xy.shape[0] > 0:
        frame_pts, status, err = cv.calcOpticalFlowPyrLK(bk_composite_frame, input_frame, fast_pts_xy, None)
        transform_matrix, status = cv.findHomography(fast_pts_xy,frame_pts,cv.RANSAC)
        warped_frame = cv.warpPerspective(input_frame, transform_matrix, (bk_composite_frame.shape[1],bk_composite_frame.shape[0]))
        return warped_frame
    else:
        return np.zeros(input_frame.shape)
def frame_extractor(input_frame,folder_name,write_frame_enable):
    global save_frame_count
    if write_frame_enable == True:
        if os.path.isdir("./" + folder_name):
            cv.imwrite("./" + folder_name + "/" + "test_frame_"+str(save_frame_count) + ".png",input_frame)
            #print("./" + folder_name + "/" + "test_frame_"+str(save_frame_count))
        else:
            os.mkdir("./" + folder_name)
            cv.imwrite("./" + folder_name + "/" + "test_frame_"+str(save_frame_count) + ".png",input_frame)
            #print("make folder")
            #print("./" + folder_name + "/" + "test_frame_"+str(save_frame_count))
        save_frame_count += 1

#------------------------------------------------------------------------------
#--Parameter Functions
#------------------------------------------------------------------------------

def input_video_process(input,frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable):

    cap = cv.VideoCapture(input)
    frame_counter = 0
#    fourcc = cap.get(cv.CAP_PROP_FOURCC)
    fps = cap.get(cv.CAP_PROP_FPS)
#    orig_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
#    orig_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    bk_remove = bk_MOG_init()

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break

        if bk_enable == True:
            bk_mask, bk_sum, frame = bk_MOG_update_v2(bk_remove,frame)


        if frame_counter == frame_skip:
            if ord('q') == cv.waitKey(int(m.ceil(1000/fps))):
                break

            gray_bk_sum = cv.cvtColor(bk_sum,cv.COLOR_BGR2GRAY)
            gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            small_frame = cv.resize(gray_frame,(width,height),interpolation = cv.INTER_AREA)

            gray_ffd_image, fast_pts_xy = fastfeaturedetect(gray_bk_sum)
            frame_warped = realtime_frame_mapper(gray_frame, gray_bk_sum, fast_pts_xy)
            cv.imshow("frame_warped", frame_warped)
            cv.imshow("ffd",gray_ffd_image)

            cv.imshow("bk_frame", gray_bk_sum)

            cv.imshow("source",small_frame)
            frame_extractor(small_frame,folder_name,write_frame_enable)
            frame_counter = 0

        frame_counter += 1

def input_camera_process(frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable):

    cap = cv.VideoCapture(0)
    frame_counter = 0
#    fourcc = cap.get(cv.CAP_PROP_FOURCC)
    fps = cap.get(cv.CAP_PROP_FPS)
#    orig_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
#    orig_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
#    print(fps)

    bk_remove = bk_MOG_init()

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break

        if bk_enable == True:
            bk_mask, bk_sum, frame = bk_MOG_update_v2(bk_remove,frame)

        if frame_counter == frame_skip:
            if ord('q') == cv.waitKey(int(m.ceil(1000/fps))):
                break
            #add your function here

            gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            small_frame = cv.resize(gray_frame,(width,height),interpolation = cv.INTER_AREA)

            if bk_enable == True:
                gray_bk_sum = cv.cvtColor(bk_sum,cv.COLOR_BGR2GRAY)
                cv.imshow("bk_frame", gray_bk_sum)
                cv.imshow("bk_r",bk_mask)

            if stable_enable == True:
                gray_ffd_image, fast_pts_xy = fastfeaturedetect(gray_bk_sum)
                frame_warped = realtime_frame_mapper(gray_frame, gray_bk_sum, fast_pts_xy)
                cv.imshow("frame_warped", frame_warped)
                cv.imshow("ffd",gray_ffd_image)

            cv.imshow("source",small_frame)

            frame_extractor(small_frame,folder_name,write_frame_enable)
            frame_counter = 0

        frame_counter += 1


#-------------------------------------------------------------------------------
#--main
# options:
# select: --view --generate
# input_type: --video --camera
# input_file == file name
# frame_skip = number of frames to remove after a displayed frame
#-------------------------------------------------------------------------------

select = "--generate" #sys.argv[1]
input_type = "--video" #sys.argv[2]
input_file = 'WIN_20210113_22_49_37_Pro.mp4' #sys.argv[3]
frame_skip = 2 #int(sys.argv[4])
width = 200 #int(sys.argv[5])
height = 200 #int(sys.argv[6])
folder_name = "gwen_t" #sys.argv[7]

bk_enable = False
stable_enable = False
write_frame_enable = False


global save_frame_count
save_frame_count = 0

if select == "--view":
    if input_type == "--video":
        input_video_process(input_file,frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable)
    elif input_type == "--camera":
        input_camera_process(frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable)
    else:
        print("type flag error")

elif select == "--viewbk":
    bk_enable = True
    if input_type == "--video":
        input_video_process(input_file,frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable)
    elif input_type == "--camera":
        input_camera_process(frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable)
    else:
        print("type flag error")

elif select == "--viewbks":
    bk_enable = True
    stable_enable = True
    if input_type == "--video":
        input_video_process(input_file,frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable)
    elif input_type == "--camera":
        input_camera_process(frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable)
    else:
        print("type flag error")

elif select == "--generate":
    write_frame_enable = True
    if input_type == "--video":
        input_video_process(input_file,frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable)
    elif input_type == "--camera":
        input_camera_process(frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable)
    else:
        print("type flag error")

elif select == "--generatebk":
    bk_enable = True
    write_frame_enable = True
    if input_type == "--video":
        input_video_process(input_file,frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable)
    elif input_type == "--camera":
        input_camera_process(frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable)
    else:
        print("type flag error")

elif select == "--generatebks":
    bk_enable = True
    stable_enable = True
    write_frame_enable = True
    if input_type == "--video":
        input_video_process(input_file,frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable)
    elif input_type == "--camera":
        input_camera_process(frame_skip,width,height,folder_name,bk_enable,stable_enable,write_frame_enable)
    else:
        print("type flag error")
else:
    print("type flag error")

sys.exit(1)
