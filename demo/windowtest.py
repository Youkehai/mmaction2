import os
import shutil

import mmcv
from mmaction.apis import init_recognizer, inference_recognizer
import cv2
from operator import itemgetter

def segment_video(video_path, segment_length):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_frames = int(fps * segment_length)

    segments = []
    for start_frame in range(0, total_frames, segment_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        segment = []
        for _ in range(segment_frames):
            ret, frame = cap.read()
            if not ret:
                break
            segment.append(frame)
        segments.append(segment)
    cap.release()
    return segments, fps

def detect_actions_in_segments(model, segments, fps):
    results = []
    for i, segment in enumerate(segments):
        temp_dir = f'segment_{i}'
        os.makedirs(temp_dir, exist_ok=True)
        for j, frame in enumerate(segment):
            frame_path = os.path.join(temp_dir, f'{j:06d}.jpg')  # 使用六位数字命名
            cv2.imwrite(frame_path, frame)
        temp_video_path = f'segment_{i}.mp4'
        mmcv.frames2video(temp_dir, temp_video_path, fps=fps)
        cur_result = inference_recognizer(model, temp_video_path)
        results.append((i, cur_result))
        os.remove(temp_video_path)
        shutil.rmtree(temp_dir)  # 使用shutil删除目录及其内容
    return results

# 初始化模型
config_file = 'work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
checkpoint_file = 'work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/best_acc_top1_epoch_4.pth'
device = 'cuda:0'
model = init_recognizer(config_file, checkpoint_file, device=device)
label = 'data/label/mylabel.txt'  # 400行的文件，每行就是数据集的一个类别

# 分割视频
video_path = 'data/two_minutes.mp4'
segment_length = 5  # 1 second per segment
segments, fps = segment_video(video_path, segment_length)

# 检测动作
results = detect_actions_in_segments(model, segments, fps)

# 输出结果
for segment_idx, result in results:
    pred_scores = result.pred_score.tolist()
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top5_label = score_sorted[:5]

    labels = open(label).readlines()
    labels = [x.strip() for x in labels]
    label_results = [(labels[k[0]], k[1]) for k in top5_label]
    start_time = segment_idx * segment_length
    end_time = start_time + segment_length
    # 查看推理Top-5结果
    for label_result in label_results:
        # if label_result[1] < 0.3:
        #     break
        print(f"Action: {label_result[0]}, Score: {label_result[1]}, Time: {start_time}-{end_time}")
