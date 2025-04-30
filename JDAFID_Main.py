
import threading
import queue
import torch
import time
import cv2
import numpy as np

from DepthAnything.depth_anything_v2.dpt import DepthAnythingV2

frame_queue = queue.Queue()
display_queue = queue.Queue()
display_queueorg = queue.Queue()

###INPUTS###
video_path = '57.mp4'  # Replace with the path to your .mp4 file
beta = 0.5 #default beta
buf_size = 10 #num of frames


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(DEVICE)
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'DepthAnything/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

def dehaze(img):
    return cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)



def process(frames,beta):
    F_list = []
    T_list = []
    A_list = []
    for i in range(0,len(frames)):

        # Convert to grayscale for processing (optional)
        #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result1 = dehaze(frames[i])

        depth = model.infer_image(result1[:,:,0])  # HxW raw depth map in numpy

        depth = 1 - ((depth - np.min(depth)) / (np.max(depth) - np.min(depth)))

        T = np.exp(-1 * depth * beta)
        frame_nrm = frames[i].astype(np.float32) / 255.0
        # Apply the DCP method to estimate I, A, and T
        T_exp = np.repeat(T[:, :, np.newaxis], 3, axis=2)
        T_list.append(T_exp)
        F_list.append(frame_nrm)
        A = (frame_nrm-result1*T_exp)/(1-T_exp+10e20)
        A = (A - np.min(A)) / (np.max(A) - np.min(A))

        A_list.append(A)


    A_array= np.array(A_list)
    T_array = np.array(T_list)
    F_array = np.array(F_list)
    IMGall = (F_array - A_array * (1 - T_array)) / T_array
    return IMGall





def camera_reader():
    cap = cv2.VideoCapture(video_path)
    buffer = []
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (480, 360))
        if not ret:
            break
        buffer.append(frame)
        if len(buffer) == buf_size:
            frame_queue.put(buffer.copy())
            buffer.clear()
    cap.release()

def frame_processor():
    while True:
        frames = frame_queue.get()

        beta_val = cv2.getTrackbarPos("Beta x10", "Controls") / 10.0
        processed_frames = process(frames, beta_val)  # beta is dynamic
        processed_frames = np.clip(processed_frames, a_min=0, a_max=1)
        display_queue.put(processed_frames)
        display_queueorg.put(frames)

def frame_displayer():
    # Create Trackbar Window
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 300, 50)
    cv2.createTrackbar("Beta x10", "Controls", int(beta*10), 15, lambda x: None)
    last_frame = None
    while True:
        try:
            frames = display_queue.get(timeout=0.1)
            framesorg = display_queueorg.get(timeout=0.1)

            for k in range(0,len(frames)):
                last_frame = frames[k]
                last_frameorg = framesorg[k]
                cv2.imshow("Processed", frames[k])
                cv2.imshow("Org", framesorg[k])
                if cv2.waitKey(100) & 0xFF == ord('q'):  # 25 FPS ~ 40 ms
                    return
        except queue.Empty:
            # Show last frame if queue is empty
            if last_frame is not None:
                #print("bossss")
                cv2.imshow("Processed", last_frame)
                cv2.imshow("Org", last_frameorg)
                if cv2.waitKey(40) & 0xFF == ord('q'):
                    return

# Thread başlatma
threads = [
    threading.Thread(target=camera_reader, daemon=True),
    threading.Thread(target=frame_processor, daemon=True),
    threading.Thread(target=frame_displayer, daemon=True),
]

for t in threads:
    t.start()

while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        break
cv2.destroyAllWindows()
