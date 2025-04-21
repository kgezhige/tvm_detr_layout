import torch, os
import numpy as np
from transformers import AutoImageProcessor
from transformers.models.detr import DetrForSegmentation
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops import nms
import onnxruntime as ort
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import onnx
# 设置设备
device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载图像处理器和模型
try:
    img_proc = AutoImageProcessor.from_pretrained("./")
    model = DetrForSegmentation.from_pretrained("./")
    model.to(device)
    model.eval()
    print('load ok')
except Exception as e:
    print(f"加载模型或处理器失败: {e}")
    exit(1)

# 加载图像
try:
    img = Image.open("3.png").convert("RGB")
except Exception as e:
    print(f"加载图像失败: {e}")
    exit(1)

# 准备输入
input_ids = img_proc(img, return_tensors="pt")
pixel_values = input_ids["pixel_values"].to(device)  # [1, 3, H, W]

# 导出 ONNX 模型
onnx_path = "detr_layout_detection.onnx"
try:
    torch.onnx.export(
        model,
        pixel_values,
        onnx_path,
        # opset_version=12,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"}
        }
    )
    print(f"模型已导出为 ONNX: {onnx_path}")
except Exception as e:
    print(f"导出 ONNX 失败: {e}")
    exit(1)

try:
    import onnx
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print("ONNX 模型检查通过")
except Exception as e:
    print(f"ONNX 模型检查失败: {e}")
    exit(1)

# ONNX 推理（仅使用 CPU）
try:
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"ONNX 推理会话创建成功，使用提供者: {ort_session.get_providers()}")
except Exception as e:
    print(f"创建 ONNX 推理会话失败: {e}")
    exit(1)

# 准备输入数据
ort_inputs = {"pixel_values": pixel_values.cpu().numpy()}
ort_outputs = ort_session.run(None, ort_inputs)

# 提取 ONNX 输出
logits = torch.tensor(ort_outputs[0])  # [batch, num_queries, num_classes]
pred_boxes = torch.tensor(ort_outputs[1])  # [batch, num_queries, 4]

# 构造后处理输出
outputs = DetrObjectDetectionOutput(
    logits=logits.to(device),
    pred_boxes=pred_boxes.to(device)
)

# 后处理：目标检测
threshold = 0.4
bbox_pred = img_proc.post_process_object_detection(
    outputs,
    threshold=threshold,
    target_sizes=[img.size[::-1]]
)

# 可视化边界框
img_np = np.array(img)
plt.figure(figsize=(10, 10))
plt.imshow(img_np)
for pred in bbox_pred:
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
for pred in bbox_pred:
    print("检测到的边界框和类别：")
    for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
        label_name = model.config.id2label.get(int(label), "Unknown")
        print(f"边界框: {box.tolist()}, 类别: {label_name}, 置信度: {score:.2f}")