import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np

# TensorRT and CUDA imports
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

class DepthEstimator:
    def __init__(
        self,
        encoder: str = 'vits',
        device: str = None,
        metric: bool = False,
        engine_path: str = None,
    ):
        # Determine device
        self.DEVICE = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_trt = engine_path is not None

        # Load TensorRT engine if provided
        if self.use_trt:
            print("Using tr")
            from depth_estimation.depth_anything_v2.dpt_tr import DepthAnythingV2
            self._load_trt_engine(engine_path)
        else:
            print("Using DepthAnythingV2")
            from depth_estimation.depth_anything_v2.dpt import DepthAnythingV2
            # Load PyTorch model
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512,1024,1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536,1536,1536,1536]},
            }
            if metric:
                from depth_estimation.metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as DV2
                max_depth = 10
                dataset   = 'hypersim'
                cfg       = {**model_configs[encoder], 'max_depth': max_depth}
                self.model = DV2(**cfg)
                ckpt_path = f'monocular_obstacle_avoidance/modules/depth_estimation/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth'
            else:
                DV2       = DepthAnythingV2
                self.model = DV2(**model_configs[encoder])
                ckpt_path = f'monocular_obstacle_avoidance/modules/depth_estimation/checkpoints/depth_anything_v2_{encoder}.pth'

            # Load weights and move to device
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location='cpu', weights_only=True)
            )
            self.model = self.model.to(self.DEVICE).eval()

    def _load_trt_engine(self, engine_path: str):
        """Deserialize a TensorRT engine and prepare buffers and context."""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Get I/O shapes
        in_shape  = self.context.get_tensor_shape('input')
        out_shape = self.context.get_tensor_shape('output')
        _, _, self.in_h, self.in_w = in_shape

        # Allocate host and device buffers
        self.h_input  = cuda.pagelocked_empty(trt.volume(in_shape),  dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(out_shape), dtype=np.float32)
        self.d_input  = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream   = cuda.Stream()

    def predict_depth(self, image: np.ndarray) -> np.ndarray:
        """Run depth prediction on a BGR image, using TRT if available, else PyTorch."""
        if self.use_trt:
            return self._predict_trt(image)
        else:
            # Use DepthAnythingV2's built-in inference (handles preprocessing internally)
            return self.model.infer_image(image)

    def _predict_trt(self, image: np.ndarray) -> np.ndarray:
        # 1) Preprocess: BGR→RGB, normalize to [0,1], resize to engine input
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)
        img = img.transpose(2, 0, 1)  # HWC→CHW

        # 2) Copy to host buffer
        np.copyto(self.h_input, img.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        # 3) Bind and run TRT inference
        self.context.set_tensor_address('input',  int(self.d_input))
        self.context.set_tensor_address('output', int(self.d_output))
        self.context.execute_async_v3(self.stream.handle)

        # 4) Retrieve output
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        depth = self.h_output.reshape(self.in_h, self.in_w)
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        # 5) Resize back to original image size
        H, W = image.shape[:2]
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_AREA)
        return depth
