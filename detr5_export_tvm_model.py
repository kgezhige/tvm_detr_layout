import time
import tvm
from tvm import relay
from tvm.relay import transform
import onnx
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 加载 ONNX 模型
start_time = time.perf_counter()
try:
    onnx_model = onnx.load("detr_layout_detection.onnx")
except Exception as e:
    print(f"加载 ONNX 模型失败: {e}")
    exit(1)
end_time = time.perf_counter()
print(f"Load ONNX model time: {end_time - start_time:.4f} seconds")

# 模拟输入形状
input_shape = [1, 3, 1024, 1024]  # [batch, channels, height, width]
shape_dict = {"pixel_values": input_shape}

# 转换为 TVM Relay IR
start_time = time.perf_counter()
try:
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    mod = transform.InferType()(mod)
except Exception as e:
    print(f"转换 ONNX 到 Relay IR 失败: {e}")
    exit(1)
end_time = time.perf_counter()
print(f"Convert ONNX to Relay IR time: {end_time - start_time:.4f} seconds")

# 保存模型和参数
prefix = "detr5_model"
with open(f"{prefix}.json", "w") as f:
    json_str = tvm.ir.save_json(mod)
    f.write(json_str)

with open(f"{prefix}.params", "wb") as f:
    f.write(relay.save_param_dict(params))

print(f"模型已保存: {prefix}.json, {prefix}.params")