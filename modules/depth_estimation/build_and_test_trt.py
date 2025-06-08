import os
import argparse

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

from depth_anything_v2.dpt_tr import load_image

def build_engine(onnx_path: str, engine_path: str, fp16: bool = False):
    """Load an ONNX model, build a TRT engine, and serialize it."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(TRT_LOGGER)
    network    = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser     = trt.OnnxParser(network, TRT_LOGGER)

    # 1) Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX model:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    # 2) Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
    trt.MemoryPoolType.WORKSPACE,
    1 << 30) # 1 GiB    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 3) (Optional) dynamic shapes profile
    profile = builder.create_optimization_profile()
    profile.set_shape("input",
                      min     =(1, 3, 518, 518),
                      opt     =(1, 3, 518, 518),
                      max     =(1, 3, 518, 518))
    config.add_optimization_profile(profile)

    # 4) Build & serialize
    print("🔨 Building TensorRT engine (this can take a while)…")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("Engine build failed")
    with open(engine_path, 'wb') as f:
        f.write(engine)
    print(f"Engine saved to {engine_path}")
    return engine

def infer_engine(engine_path: str,
                 img_path:    str,
                 outdir:      str,
                 grayscale:   bool = False):
    """Load TRT engine from disk, run one inference on img_path, save depth map."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # 1) Deserialize engine
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # 2) Prepare I/O buffers
    input_image, (orig_h, orig_w) = load_image(img_path)
    input_shape  = context.get_tensor_shape('input')
    output_shape = context.get_tensor_shape('output')

    h_input  = cuda.pagelocked_empty(trt.volume(input_shape),  dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_input  = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream   = cuda.Stream()

    # 3) Copy in
    np.copyto(h_input, input_image.ravel())
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # 4) Bind device buffers to tensor names
    context.set_tensor_address("input", int(d_input))
    context.set_tensor_address("output", int(d_output))

    # 5) Run inference asynchronously
    context.execute_async_v3(stream.handle)

    # 6) Copy out
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    # 7) Post-process & save
    depth = h_output.reshape(output_shape[2:])
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = cv2.resize(depth, (orig_w, orig_h))

    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(outdir, f"{base}_depth.png")

    if grayscale:
        cv2.imwrite(out_path, depth)
    else:
        colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(out_path, colored)

    print(f"Inference complete — depth map saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a TRT engine from ONNX and run a test inference"
    )
    parser.add_argument("--onnx",    required=True,
                        help="Path to the .onnx model")
    parser.add_argument("--engine",  required=True,
                        help="Path where to save / load the engine (.trt or .engine)")
    parser.add_argument("--img",     required=True,
                        help="Path to a test image (e.g. foto.png)")
    parser.add_argument("--outdir",  default="./vis_depth",
                        help="Directory for output depth map")
    parser.add_argument("--fp16",    action="store_true",
                        help="Build engine in fp16 mode")
    parser.add_argument("--grayscale", action="store_true",
                        help="Save depth map as grayscale")
    args = parser.parse_args()

    # Only build if engine does not already exist
    if not os.path.exists(args.engine):
        build_engine(args.onnx, args.engine, fp16=args.fp16)
    else:
        print(f"► Engine '{args.engine}' already exists; skipping build.")

    # Run inference
    infer_engine(args.engine, args.img, args.outdir, grayscale=args.grayscale)
