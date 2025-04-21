import time
import tvm
from tvm import relay, autotvm
from tvm.relay import transform
import tvm.runtime
from tvm.contrib import graph_executor
import onnx
import numpy as np
from transformers import AutoImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 设置目标（可通过变量选择）
target = "llvm"  # 选项：llvm, cuda, opencl

# 目标到设备的映射
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

# 加载图像处理器
start_time = time.perf_counter()
try:
    img_proc = AutoImageProcessor.from_pretrained("./")
except Exception as e:
    print(f"加载图像处理器失败: {e}")
    exit(1)
end_time = time.perf_counter()
print(f"Load image processor time: {end_time - start_time:.4f} seconds")

# 加载图像
start_time = time.perf_counter()
try:
    img = Image.open("./Pics/ta3.jpeg").convert("RGB")
except Exception as e:
    print(f"加载图像失败: {e}")
    exit(1)
end_time = time.perf_counter()
print(f"Load image time: {end_time - start_time:.4f} seconds")

# 准备输入
start_time = time.perf_counter()
input_ids = img_proc(img, return_tensors="np")  # 使用 NumPy 后端
pixel_values = input_ids["pixel_values"]  # [1, 3, H, W]
end_time = time.perf_counter()
print(f"Preprocess image time: {end_time - start_time:.4f} seconds")

# 打印输入格式
print("输入键：", list(input_ids.keys()))
for key, value in input_ids.items():
    print(f"{key}: 形状={value.shape}, 类型={value.dtype}")
print(f"图像处理器配置: {img_proc}")

# 加载 ONNX 模型
start_time = time.perf_counter()
try:
    onnx_model = onnx.load("detr_layout_detection.onnx")
except Exception as e:
    print(f"加载 ONNX 模型失败: {e}")
    exit(1)
end_time = time.perf_counter()
print(f"Load ONNX model time: {end_time - start_time:.4f} seconds")

# 转换为 TVM Relay IR
input_shape = pixel_values.shape  # 例如，[1, 3, 800, 1066]
shape_dict = {"pixel_values": input_shape}
start_time = time.perf_counter()
try:
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    mod = transform.InferType()(mod)
except Exception as e:
    print(f"转换 ONNX 到 Relay IR 失败: {e}")
    exit(1)
end_time = time.perf_counter()
print(f"Convert ONNX to Relay IR time: {end_time - start_time:.4f} seconds")

# TVM 算子调优
tuning_log = f"tvm_tuning_{target}.log"
if not os.path.exists(tuning_log):
    print(f"运行算子调优 ({target})...")
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
    for i, task in enumerate(tasks):
        print(f"调优任务 {i+1}/{len(tasks)}: {task.name}")
        tuner = autotvm.tuner.XGBTuner(task)
        tuner.tune(
            n_trial=20,  # 调优次数（可增加）
            measure_option=autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(number=5)
            ),
            callbacks=[autotvm.callback.log_to_file(tuning_log)]
        )
    print(f"调优完成，日志保存至: {tuning_log}")

# 编译模型
start_time = time.perf_counter()
try:
    with autotvm.apply_history_best(tuning_log):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
    graph_mod = graph_executor.GraphModule(lib["default"](device))
except Exception as e:
    print(f"编译模型 ({target}) 失败: {e}")
    exit(1)
end_time = time.perf_counter()
print(f"Compile model ({target}) time: {end_time - start_time:.4f} seconds")

# 推理函数
def run_tvm_inference(graph_mod, input_data):
    start_time = time.perf_counter()
    graph_mod.set_input("pixel_values", input_data)
    graph_mod.run()
    outputs = [graph_mod.get_output(i).numpy() for i in range(2)]  # logits, pred_boxes
    end_time = time.perf_counter()
    print(f"Inference time ({target}): {end_time - start_time:.4f} seconds")
    return outputs

# 运行推理
start_time = time.perf_counter()
outputs = run_tvm_inference(graph_mod, pixel_values)
end_time = time.perf_counter()
print(f"Total inference time ({target}): {end_time - start_time:.4f} seconds")

# 后处理（修复 AttributeError）
start_time = time.perf_counter()
logits = outputs[0]  # [batch, num_queries, num_classes]
pred_boxes = outputs[1]  # [batch, num_queries, 4]

# 手动后处理（兼容不支持字典的 transformers 版本）
def custom_post_process_object_detection(logits, pred_boxes, target_sizes, threshold=0.4, num_labels=91):
    import numpy as np
    # 转换为概率
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    # 提取高置信度预测
    keep = probs.max(-1) > threshold
    scores = probs[keep]
    labels = np.argmax(probs, axis=-1)[keep]
    boxes = pred_boxes[keep]
    
    # 调整边界框到原始图像尺寸
    target_height, target_width = target_sizes[0]
    boxes = boxes.copy()
    boxes[:, [0, 2]] *= target_width  # x1, x2
    boxes[:, [1, 3]] *= target_height  # y1, y2
    
    # 构造结果
    results = [{
        "scores": scores,
        "labels": labels,
        "boxes": boxes
    }]
    return results

# 使用自定义后处理
bbox_pred = custom_post_process_object_detection(
    logits=logits,
    pred_boxes=pred_boxes,
    target_sizes=[img.size[::-1]],
    threshold=0.4,
    num_labels=len(img_proc.model_config.id2label)
)
end_time = time.perf_counter()
print(f"Postprocess time: {end_time - start_time:.4f} seconds")

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
        label_name = img_proc.model_config.id2label.get(int(label), "Unknown")
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="r", facecolor="none")
        plt.gca().add_patch(rect)
        plt.text(x1, y1-10, f"{label_name} ({score:.2f})", color="r", fontsize=12, weight="bold")
plt.axis("off")
plt.show()

# 打印边界框和类别
print(f"检测到的边界框和类别 ({target})：")
for pred in bbox_pred:
    for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
        label_name = img_proc.model_config.id2label.get(int(label), "Unknown")
        print(f"边界框: {box.tolist()}, 类别: {label_name}, 置信度: {score:.2f}")