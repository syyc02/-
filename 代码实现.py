import cv2
import mediapipe as mp

# 1. 初始化姿态检测模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                    min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 2. 读取视频流
cap = cv2.VideoCapture(0) # 调用默认摄像头

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # OpenCV 默认 BGR，转换为模型需要的 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. 执行检测
    results = pose.process(image_rgb)
    
    # 4. 绘制结果
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow('Pose Detection Demo', image)
    if cv2.waitKey(5) & 0xFF == 27: # 按 ESC 退出
        break

cap.release()
cv2.destroyAllWindows()