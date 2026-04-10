import tensorrt as trt
import numpy as np
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path, use_fp16=False):
    print(f"Building TensorRT engine: {engine_file_path}")
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        config = builder.create_builder_config()
        if use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled")

        with open(onnx_file_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return None

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Failed to build engine")
            return None

        with open(engine_file_path, 'wb') as f:
            f.write(serialized_engine)
        print(f"Engine saved to {engine_file_path}")
        return engine_file_path

if __name__ == "__main__":
    ONNX_PATH = r"F:\python_code\pt_to_onnx\model_pruned.onnx"   # 改为你的ONNX路径
    ENGINE_PATH = r"F:\python_code\pt_to_onnx\model_fp16.engine"
    if not os.path.exists(ONNX_PATH):
        print(f"ONNX file not found: {ONNX_PATH}")
    else:
        build_engine(ONNX_PATH, ENGINE_PATH, use_fp16=True)