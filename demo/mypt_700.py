from mmaction.apis import inference_skeleton, init_recognizer
from operator import itemgetter



config_path = 'configs/skeleton/2s-agcn/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py'
#checkpoint_path = 'work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/best_acc_top1_epoch_1.pth'  # 可以是本地路径
checkpoint_path = 'models/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221222-4c0ed77e.pth'  # 可以是本地路径
img_path = 'data/drink_beer_video.mp4'  # 您可以指定自己的图片路径

label = 'tools/data/kinetics710/label_map_k710.txt'  # 400行的文件，每行就是数据集的一个类别
# 从配置文件和权重文件中构建模型
model = init_recognizer(config_path, checkpoint_path, device='cuda:0')  # device 可以是 'cuda:0'
# 对单个视频进行测试
pred_result = inference_skeleton(model, img_path)


pred_scores = pred_result.pred_score.tolist()
score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
top5_label = score_sorted[:5]

labels = open(label).readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in top5_label]
# 查看推理Top-5结果
for result in results:
    print(f'{result[0]}: ', result[1])
