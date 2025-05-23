import time
import tvm
from tvm import relay, autotvm
from tvm.relay import transform
from tvm.contrib import graph_executor
import numpy as np
from transformers import AutoImageProcessor, DetrConfig
from PIL import Image
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 优化开关
enable_fusion = False
enable_quantization = False  # 与 export_tvm_model.py 一致
enable_autotune = True

# 设置目标
target = "cuda"  # 选项：llvm, cuda, opencl
target_device_map = {
    "llvm": tvm.cpu(0),
    "cuda": tvm.cuda(0),
    "opencl": tvm.opencl(0)
}

if target not in target_device_map:
    print(f"不支持的目标: {target}. 可用目标: {list(target_device_map.keys())}")
    exit(1)

device = target_device_map[target]
print(f"使用目标: {target}, 设备: {device}")

# 加载图像处理器和配置
start_time = time.perf_counter()
try:
    img_proc = AutoImageProcessor.from_pretrained("./")
    config = DetrConfig.from_pretrained("./")
    id2label = config.id2label
    num_labels = len(id2label)
except Exception as e:
    print(f"加载图像处理器或配置失败: {e}")
    id2label = {
        0: "Caption", 1: "Footnote", 2: "Formula", 3: "List-item", 4: "Page-footer",
        5: "Page-header", 6: "Picture", 7: "Section-header", 8: "Table", 9: "Text", 10: "Title"
    }
    num_labels = len(id2label)
    print("使用硬编码 id2label")
end_time = time.perf_counter()
print(f"Load image processor and config time: {end_time - start_time:.4f} seconds")
print(f"labels: {id2label}")

# 加载图像
start_time = time.perf_counter()
try:
    img = Image.open("./Pics/3.png").convert("RGB")
except Exception as e:
    print(f"加载图像失败: {e}")
    exit(1)
end_time = time.perf_counter()
print(f"Load image time: {end_time - start_time:.4f} seconds")

# 准备输入
start_time = time.perf_counter()
input_ids = img_proc(
    img,
    return_tensors="np",
    # size={"shortest_edge": 800, "longest_edge": 1333}
)
pixel_values = input_ids["pixel_values"]
if enable_quantization:
    pixel_values = pixel_values.astype(np.int8)  # 量化输入（简单模拟）
end_time = time.perf_counter()
print(f"Preprocess image time: {end_time - start_time:.4f} seconds")
print(f"Image size: {img.size}, pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")

# 加载 TVM 模型和参数
start_time = time.perf_counter()
try:
    with open("detr5_model.json", "r") as f:
        mod = tvm.ir.load_json(f.read())
    with open("detr5_model.params", "rb") as f:
        params = relay.load_param_dict(f.read())
except Exception as e:
    print(f"加载模型或参数失败: {e}")
    exit(1)
end_time = time.perf_counter()
print(f"Load model and params time: {end_time - start_time:.4f} seconds")

# 统计参数量和数据类型
def get_model_stats(mod, params):
    param_count = 0
    dtypes = set()
    for k, v in params.items():
        param_count += np.prod(v.shape)
        dtypes.add(str(v.dtype))
    return param_count, dtypes

param_count, dtypes = get_model_stats(mod, params)
print(f"加载模型参数量: {param_count}, 数据类型: {dtypes}")

# 算子调优
tuning_log = f"tvm_tuning_{target}.log"
if enable_autotune and (not os.path.exists(tuning_log) or os.path.getsize(tuning_log) == 0):
    print(f"调优日志不存在或为空，运行算子调优 ({target})...")
    start_time = time.perf_counter()
    try:
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
        for i, task in enumerate(tasks):
            print(f"调优任务 {i+1}/{len(tasks)}: {task.name}")

            tuner = autotvm.tuner.GridSearchTuner(task)
            tuner.tune(
                n_trial=5,  # 限制尝试次数，避免无限搜索
                measure_option=autotvm.measure_option(
                    builder=autotvm.LocalBuilder(),
                    runner=autotvm.LocalRunner(number=3)
                ),
                callbacks=[autotvm.callback.log_to_file(tuning_log)]
            )
    except Exception as e:
        print(f"算子调优失败 ({target}): {e}")
        print("继续编译，不使用调优日志")
        tuning_log = None
    end_time = time.perf_counter()
    print(f"Tuning time ({target}): {end_time - start_time:.4f} seconds")
elif enable_autotune:
    print(f"使用现有调优日志: {tuning_log}")
else:
    print("自动调优已禁用")
    tuning_log = None

# 编译模型
start_time = time.perf_counter()
try:
    if tuning_log:
        with autotvm.apply_history_best(tuning_log):
            with tvm.transform.PassContext(opt_level=0):
                lib = relay.build(mod, target=target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(mod, target=target, params=params)
    graph_mod = graph_executor.GraphModule(lib["default"](device))
except Exception as e:
    print(f"编译模型 ({target}) 失败: {e}")
    exit(1)
end_time = time.perf_counter()
print(f"Compile model ({target}) time: {end_time - start_time:.4f} seconds")

# 编译模型
# start_time = time.perf_counter()
# try:
#     with tvm.transform.PassContext(opt_level=0):
#         lib = relay.build(mod, target=target, params=params)
#     graph_mod = graph_executor.GraphModule(lib["default"](device))
# except Exception as e:
#     print(f"编译模型 ({target}) 失败: {e}")
#     exit(1)
# end_time = time.perf_counter()
# print(f"Compile model ({target}) time: {end_time - start_time:.4f} seconds")


# 统计编译后文件大小
lib.export_library("detr_model.so")
so_size = os.path.getsize("detr_model.so") / 1024 / 1024  # MB
print(f"编译模型大小: detr_model.so ({so_size:.2f} MB)")

# 推理函数
def run_tvm_inference(graph_mod, input_data):
    start_time = time.perf_counter()
    try:
        graph_mod.set_input("pixel_values", input_data)
        graph_mod.run()
        outputs = [graph_mod.get_output(i).numpy() for i in range(2)]
    except Exception as e:
        print(f"推理失败: {e}")
        return None
    end_time = time.perf_counter()
    print(f"Inference time ({target}): {end_time - start_time:.4f} seconds")
    print(f"logits shape: {outputs[0].shape}, sample: {outputs[0][0, :5, :3]}")
    print(f"pred_boxes shape: {outputs[1].shape}, sample: {outputs[1][0, :5]}")
    return outputs

# 运行推理
start_time = time.perf_counter()
outputs = run_tvm_inference(graph_mod, pixel_values)
if outputs is None:
    print("推理无结果，退出")
    exit(1)
end_time = time.perf_counter()
print(f"Total inference time ({target}): {end_time - start_time:.4f} seconds")

# 后处理
start_time = time.perf_counter()
logits = outputs[0]
pred_boxes = outputs[1]

def nms(boxes, scores, iou_threshold=0.5):
    """非极大值抑制"""
    if boxes.size == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = w * h / (areas[i] + areas[order[1:]] - w * h + 1e-10)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)

def custom_post_process_object_detection(logits, pred_boxes, target_sizes, threshold=0.5, num_labels=12):
    import numpy as np
    # 转换为概率
    probs = 1 / (1 + np.exp(-logits))  # (1, 100, 12)
    print(f"probs max: {probs.max(-1).max()}, min: {probs.max(-1).min()}")
    
    # 提取高置信度预测
    keep = probs.max(-1) > threshold  # (1, 100)
    scores = probs.max(-1)[keep].flatten()  # 最大概率
    labels = np.argmax(probs, axis=-1)[keep]  # 类别索引
    
    # 过滤背景类（索引 11）
    valid = labels != 11
    scores = scores[valid]
    labels = labels[valid]
    boxes = pred_boxes[keep][valid]
    
    # 转换为角落坐标
    target_height, target_width = target_sizes[0]
    if boxes.size > 0:  # 确保非空
        center_x, center_y, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (center_x - width / 2) * target_width
        y1 = (center_y - height / 2) * target_height
        x2 = (center_x + width / 2) * target_width
        y2 = (center_y + height / 2) * target_height
        boxes = np.stack([x1, y1, x2, y2], axis=-1)

        # 应用 NMS
        keep_indices = nms(boxes, scores, iou_threshold=0.5)
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        boxes = boxes[keep_indices]
    else:
        boxes = np.array([]).reshape(0, 4)
    
    print(f"scores shape: {scores.shape}, sample: {scores[:5]}")
    print(f"labels shape: {labels.shape}, sample: {labels[:5]}")
    print(f"boxes shape: {boxes.shape}, sample: {boxes[:5]}")
    
    results = [{
        "scores": scores,
        "labels": labels,
        "boxes": boxes
    }]
    return results

bbox_pred = custom_post_process_object_detection(
    logits=logits,
    pred_boxes=pred_boxes,
    target_sizes=[img.size[::-1]],
    threshold=0.5,
    num_labels=num_labels
)
end_time = time.perf_counter()
print(f"Postprocess time: {end_time - start_time:.4f} seconds")
print(f"bbox_pred: {len(bbox_pred)}, boxes: {[pred['boxes'].shape for pred in bbox_pred]}")

# 可视化边界框
img_np = np.array(img)
plt.figure(figsize=(10, 10))
plt.title(f"边界框检测结果 ({target})")
plt.imshow(img_np)
for pred in bbox_pred:
    boxes = pred["boxes"]
    scores = pred["scores"]
    labels = pred["labels"]
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        label_name = id2label.get(int(label), "Unknown")
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="r", facecolor="none")
        plt.gca().add_patch(rect)
        plt.text(x1, y1-10, f"{label_name} ({score.item():.2f})", color="r", fontsize=12, weight="bold")
        print(f"box: {box}, score: {score}, label: {label_name}")
plt.axis("off")
plt.show()

# 打印边界框和类别
print(f"检测到的边界框和类别 ({target})：")
for pred in bbox_pred:
    for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
        label_name = id2label.get(int(label), "Unknown")
        print(f"边界框: {box.tolist()}, 类别: {label_name}, 置信度: {score.item():.2f}")