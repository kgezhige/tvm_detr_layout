import time
import tvm
from tvm import relay
from tvm.relay import transform
import onnx
import os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 优化开关
enable_fusion = False
enable_quantization = False  # INT8 需要校准，设为 False 使用 FP16 伪量化
enable_autotune = False

# 加载 ONNX 模型
start_time = time.perf_counter()
try:
    onnx_model = onnx.load("detr_layout_detection.onnx")
    print("ONNX 模型加载成功")
except Exception as e:
    print(f"加载 ONNX 模型失败: {e}")
    exit(1)
end_time = time.perf_counter()
print(f"Load ONNX model time: {end_time - start_time:.4f} seconds")

# 定义输入形状
shape_dict = {"pixel_values": [1, 3, 1024, 1024]}

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

# 优化：算子融合
if enable_fusion:
    print("应用算子融合...")
    mod = transform.FuseOps(fuse_opt_level=1)(mod)  #
    # mod = transform.InferType()(mod)

# 优化：量化（INT8 或 FP16）
if enable_quantization:
    print("应用量化（INT8）...")
    try:
        # 简单 INT8 量化（无校准数据集）
        with tvm.transform.PassContext(opt_level=0):
            mod = relay.quantize.quantize(
                mod,
                params=params,
                dataset=None,  # 无校准数据，使用默认配置
                # target_dtype="int8"
            )
        params = relay.quantize.preprocess_params(params, mod)
    except Exception as e:
        print(f"INT8 量化失败: {e}, 回退到 FP16")
        mod = transform.ToMixedPrecision("float16")(mod)
        params = {k: v.astype("float16") for k, v in params.items()}
else:
    print("未启用量化，使用 FP32 或 FP16 伪量化")
    # 伪量化：转换为 FP16（可选）
    if False:  # 手动启用 FP16
        mod = transform.ToMixedPrecision("float16")(mod)
        params = {k: v.astype("float16") for k, v in params.items()}

# 统计参数量和数据类型
def get_model_stats(mod, params):
    param_count = 0
    dtypes = set()
    for k, v in params.items():
        param_count += np.prod(v.shape)
        dtypes.add(str(v.dtype))
    return param_count, dtypes

param_count, dtypes = get_model_stats(mod, params)
print(f"优化后参数量: {param_count}, 数据类型: {dtypes}")

# 保存模型和参数
prefix = "detr5_model"
with open(f"{prefix}.json", "w") as f:
    json_str = tvm.ir.save_json(mod)
    f.write(json_str)

with open(f"{prefix}.params", "wb") as f:
    f.write(relay.save_param_dict(params))

# 统计文件大小
json_size = os.path.getsize(f"{prefix}.json") / 1024 / 1024  # MB
params_size = os.path.getsize(f"{prefix}.params") / 1024 / 1024  # MB
print(f"模型已保存: {prefix}.json ({json_size:.2f} MB), {prefix}.params ({params_size:.2f} MB)")