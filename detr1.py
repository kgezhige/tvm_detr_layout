import torch, os
import numpy as np
from transformers import AutoImageProcessor
from transformers.models.detr import DetrForSegmentation
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops import nms
import onnxruntime as ort
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 设置设备
device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载图像处理器和模型
try:
    img_proc = AutoImageProcessor.from_pretrained("./")
    print('impge proc', img_proc)
    model = DetrForSegmentation.from_pretrained("./")
    model.to(device)
    model.eval()
    print('load ok')
except Exception as e:
    print(f"加载模型或处理器失败: {e}")
    exit(1)

# 加载图像
try:
    img = Image.open("./Pics/ta3.jpeg").convert("RGB")
except Exception as e:
    print(f"加载图像失败: {e}")
    exit(1)

# 推理
with torch.inference_mode():
    input_ids = img_proc(img, return_tensors="pt")
    input_ids = {k: v.to(device) for k, v in input_ids.items()}
    output = model(**input_ids)

# 设置阈值
threshold = 0.4
nms_iou_threshold = 0.5  # NMS 的 IoU 阈值，调整以控制去重严格程度

# 后处理：目标检测
bbox_pred = img_proc.post_process_object_detection(
    output,
    threshold=threshold,
    target_sizes=[img.size[::-1]]
)

# 应用 NMS 去重
filtered_bbox_pred = []
for pred in bbox_pred:
    boxes = pred["boxes"]  # [N, 4]，格式 [x1, y1, x2, y2]
    scores = pred["scores"]  # [N]
    labels = pred["labels"]  # [N]
    
    # 转换为 Tensor
    boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
    scores = torch.tensor(scores, dtype=torch.float32).to(device)
    
    # 应用 NMS
    keep_indices = nms(boxes, scores, iou_threshold=nms_iou_threshold)
    
    # 过滤保留的框
    filtered_boxes = boxes[keep_indices].cpu().numpy()
    filtered_scores = scores[keep_indices].cpu().numpy()
    filtered_labels = labels[keep_indices].cpu().numpy()
    
    filtered_bbox_pred.append({
        "boxes": filtered_boxes,
        "scores": filtered_scores,
        "labels": filtered_labels
    })

# 可视化边界框
img_np = np.array(img)
plt.figure(figsize=(10, 10))
plt.imshow(img_np)
for pred in filtered_bbox_pred:
    boxes = pred["boxes"]
    scores = pred["scores"]
    labels = pred["labels"]
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        label_name = model.config.id2label.get(int(label), "Unknown")
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="r", facecolor="none")
        plt.gca().add_patch(rect)
        plt.text(x1, y1-10, f"{label_name} ({score:.2f})", color="r", fontsize=12, weight="bold")
plt.axis("off")
plt.show()

# 打印边界框和类别
for pred in filtered_bbox_pred:
    print("检测到的边界框和类别：")
    for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
        label_name = model.config.id2label.get(int(label), "Unknown")
        print(f"边界框: {box.tolist()}, 类别: {label_name}, 置信度: {score:.2f}")