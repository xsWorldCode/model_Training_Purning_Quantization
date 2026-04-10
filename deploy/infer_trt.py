import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = self.allocate_buffers()

    def load_engine(self, engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        # 遍历所有输入输出张量 (TensorRT 10.x 使用 get_tensor_name)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                outputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape})
        self.stream = stream
        return inputs, outputs, bindings

    def infer(self, input_data):
        # 将输入数据拷贝到主机内存
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        # 将数据从主机传输到设备
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        # 设置输入张量地址
        self.context.set_tensor_address(self.inputs[0]['name'], int(self.inputs[0]['device']))
        # 设置输出张量地址
        self.context.set_tensor_address(self.outputs[0]['name'], int(self.outputs[0]['device']))
        # 异步执行推理
        self.context.execute_async_v3(self.stream.handle)
        # 将结果从设备传回主机
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        output = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
        return output

def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

if __name__ == "__main__":
    ENGINE_PATH = r"F:\python_code\pt_to_onnx\model_fp16.engine"   # 你的 engine 文件路径
    IMAGE_PATH = r"F:\python_code\DataSets\split\test\Cat\cat.50.jpg"

    if not os.path.exists(ENGINE_PATH):
        print(f"Engine file not found: {ENGINE_PATH}")
        exit(1)

    trt_infer = TRTInference(ENGINE_PATH)
    input_tensor = preprocess(IMAGE_PATH)
    print(f"Input shape: {input_tensor.shape}")

    output = trt_infer.infer(input_tensor)
    print(f"Logits: {output}")
    prob = np.exp(output) / np.sum(np.exp(output))
    print(f"Probability: {prob}")
    print(f"Predicted class: {np.argmax(prob)}")