import cv2
from mmaction.apis import inference_recognizer, init_recognizer

# 配置文件和模型文件路径
config_file = 'configs/recognition/c2d/c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb.py'
checkpoint_file = 'models/c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb_20221027-5f382a43.pth'

# 初始化模型
model = init_recognizer(config_file, checkpoint_file, device='cuda:0')

# 视频路径
video_path = 'data/test.mp4'
out_video_path = 'data/output_video_with_detections.mp4'

# 读取视频
cap = cv2.VideoCapture(video_path)

# 获取视频属性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))

# 读取视频帧并进行动作检测
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 动作检测
    results = inference_recognizer(model, frame)

    # 假设results是一个包含动作类别和置信度的列表
    for result in results:
        action, confidence = result
        # 绘制红框和置信度（这里需要根据模型输出的具体格式进行调整）
        # 例如，如果动作是在图像中心发生的，可以这样绘制：
        cv2.rectangle(frame, (frame_width // 4, frame_height // 4), (3 * frame_width // 4, 3 * frame_height // 4), (0, 0, 255), 2)
        cv2.putText(frame, f'{action}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 写入帧到视频文件
    out.write(frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()