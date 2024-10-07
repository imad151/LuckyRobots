import luckyrobots as lr
import cv2
import os
'''
Results:
rgb cam resolution: 720x1280x3
same for depth cam
'''

@lr.on("robot_output")
def printoutput(msg):
    while True:
        if msg and isinstance(msg, dict) and 'rgb_cam1' in msg:
            image_path = msg['rgb_cam2'].get('file_path')
            greyscale_img = msg['depth_cam2'].get('file_path')

            if os.path.exists(image_path) and os.path.exists(greyscale_img):
                print(image_path)
                img = cv2.imread(image_path)
                greyscsale = cv2.imread(greyscale_img)
                h, w, ch = img.shape
                hg, wg, chg = greyscsale.shape
                print(f'For RGB Cam height: {h}, width: {w}, channels: {ch}')
                print(f'For depth Cam height: {hg}, width: {wg}, channels: {chg}')

                break


lr.start()