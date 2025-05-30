import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from fungsi import initialize_features
from fungsi import POS
from scipy.signal import butter, filtfilt, find_peaks

resp_signal = []
rppg_signal = []
r_signal, g_signal, b_signal = [], [], []

fps = 30
time_window = 60
frame_buffer_limit = time_window * fps
frame_buffer = []

features = None
lk_params = None
old_gray = None
left_x, top_y, right_x, bottom_y = None, None, None, None
STANDARD_SIZE = (640, 480)

margin_x = 10  
scaling_factor = 0.8

model_path = "models/pose_landmarker.task"
BaseOptions = mp.tasks.BaseOptions

PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode


## membuat landmarker untuk sinyal respirasi
options_image = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path,
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options_image)


## membuat facedetector untuk sinyal rppg

base_model="models/blaze_face_short_range.tflite"
base_options = python.BaseOptions(model_asset_path=base_model)

FaceDetectorOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode
options = FaceDetectorOptions(
    base_options=base_options,
    running_mode = VisionRunningMode.IMAGE,
)
face_detector = vision.FaceDetector.create_from_options(options)


def bandpass_filter(signal, fs, lowcut, highcut, order=3):
    nyq = 0.5 * fs  
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')  
    filtered = filtfilt(b, a, signal) 
    return filtered


file_path = os.path.join("sampel/qaessar.mp4")

try:
    cap = cv2.VideoCapture(file_path) 

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    x = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while len(frame_buffer) < frame_buffer_limit:
        x = x + 1
        ret, frame = cap.read() 
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, STANDARD_SIZE)
        frame_buffer.append(frame)

        if features is None:
            features, lk_params, old_gray, left_x, top_y, right_x, bottom_y = initialize_features(frame, pose_landmarker, STANDARD_SIZE)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(features) > 10:
            new_features, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, features, None, **lk_params)
            good_old = features[status == 1]
            good_new = new_features[status == 1]
            mask = np.zeros_like(frame)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)
            frame = cv2.add(frame, mask)
            if len(good_new) > 0:
                avg_y = np.mean(good_new[:, 1])
                resp_signal.append(avg_y)
                features = good_new.reshape(-1, 1, 2)
            old_gray = frame_gray.copy()
        else:
            initialize_features(frame, pose_landmarker, STANDARD_SIZE) ##inih

        cv2.rectangle(frame, (int(left_x), int(top_y)), (int(right_x), int(bottom_y)), (0, 255, 0), 2)

        
        print(f"Processed frame {x}/{total_frames}")

    
    x = 0

    frame_buffer = []

    cap = cv2.VideoCapture(file_path) 

    while len(frame_buffer) < frame_buffer_limit:
        x = x + 1
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_buffer.append(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = face_detector.detect(mp_image)

        if result.detections:
            for detection in result.detections:
                bboxC = detection.bounding_box
                x, y, w, h = bboxC.origin_x, bboxC.origin_y, bboxC.width, bboxC.height

                new_x = int(x + margin_x)
                new_w = int(w * scaling_factor)
                new_h = int(h * scaling_factor)

                face_roi = rgb_frame[y:y+new_h, new_x:new_x+new_w]

                mean_rgb = cv2.mean(face_roi)[:3]

                r_signal.append(mean_rgb[0])
                g_signal.append(mean_rgb[1])
                b_signal.append(mean_rgb[2])

        print(f"Processed frame {x}/{total_frames}")
    
    rgb_signals = np.array([r_signal, g_signal, b_signal])
    rgb_signals = rgb_signals.reshape(1, 3, -1)
    rppg_signal = POS(rgb_signals, fps=30)
    rppg_signal = rppg_signal.reshape(-1)

except Exception as e:
    print(f"An error occurred: {e}")
    cap.release()

finally:
    cap.release()

    print(len(rppg_signal))

    sinyal_respirasi_filter = bandpass_filter(resp_signal, 30, 0.1, 0.5)
    sinyal_rppg_filter = bandpass_filter(rppg_signal, 30, 0.7, 2.5)

    jarak_minimal = int(fps * 1.5) 
    puncak_respirasi, _ = find_peaks(sinyal_respirasi_filter)
    puncak_rppg, _ = find_peaks(sinyal_rppg_filter)


    breaths_per_minute = len(puncak_respirasi) / (len(sinyal_respirasi_filter) / 30) * 60
    heart_rate =  60 * len(puncak_rppg) / (len(sinyal_rppg_filter) / 30)
    


    ## menampilkan sinyal mentah  

    fig, ax = plt.subplots(2, 1, figsize=(40, 5))
    ax[0].plot(resp_signal, color='blue')
    ax[0].set_title('Sinyal respirasi')
    ax[1].plot(rppg_signal, color='red')
    ax[1].set_title('Sinyal rppg')
    plt.tight_layout()
    plt.show()


    ## menampilkan sinyal terfilter

    fig, ax = plt.subplots(2, 1, figsize=(40, 5))
    ax[0].plot(sinyal_respirasi_filter, color='blue')
    ax[0].set_title(f'Sinyal Respirasi, BPM: {breaths_per_minute:.2f})')
    ax[0].plot(puncak_respirasi, sinyal_respirasi_filter[puncak_respirasi], 'rx', label='Detected Breaths')

    ax[1].plot(sinyal_rppg_filter, color='red')
    ax[1].set_title(f'Sinyal RPPG, Denyut Jantung : {heart_rate:.2f})')
    ax[1].plot(puncak_rppg, sinyal_rppg_filter[puncak_rppg], 'rx', label='Detected Breaths')
    plt.tight_layout()
    plt.show()