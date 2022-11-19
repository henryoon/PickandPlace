import sys
try:
    import cv2
except:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
from calibration.calibration import HandEyeCalib

if  __name__ =="__main__":
    HandEyeCal=HandEyeCalib()

    # intrinsic parameter 불러오기
    HandEyeCal.load_CameraMatrix(filename="/mnt/workspace/roboplus_challenge/data/cameraMatrix.npy")

    RGB = cv2.imread("/mnt/workspace/roboplus_challenge/data/calibration_RGB.png")

    # 단일 이미지를 활용한 Eye in Hand Calibration
    R,Calib_result,display_img=HandEyeCal.Calibration_EyeinHand_SingleImg(RGB,flag_show=True)

    # 툴 길이 보정
    # camera to TCP (카메라 좌표계 기준)
    Calib_result[2]-=70 # unit millimeter

    # 쓰기
    HandEyeCal.write_calib(Calib_result,filename="/mnt/workspace/roboplus_challenge/calibration/calibration.json")

    # 불러오기
    Calib=HandEyeCal.load_calibration(filename="/mnt/workspace/roboplus_challenge/calibration/calibration.json")

    # 프린트
    print(Calib)