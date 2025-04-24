# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 5:20
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : get_trt.py
# ------❤❤❤------ #

import argparse
import os

import tensorrt as trt
from cuda import cudart


def convert_onnx_to_engine(args):
    _, free_mem, total_mem = cudart.cudaMemGetInfo()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(args.onnx)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        print("Failed to parse ONNX file.")
        return
    config = builder.create_builder_config()
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.max_workspace_size = total_mem // 2
    serialized_engine = builder.build_serialized_network(network, config)

    if os.path.exists(args.engine):
        overwrite = input("The engine file already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Operation aborted.")
            return

    with open(args.engine, "wb") as f:
        f.write(serialized_engine)
        print("Engine file saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX model to TensorRT engine')
    parser.add_argument('--onnx', type=str, required=True, help='Path to the ONNX model file')
    parser.add_argument('--engine', type=str, required=True, help='Path to save the TensorRT engine file')
    parser.add_argument('--fp16', type=str, default=False)

    args = parser.parse_args()

    convert_onnx_to_engine(args)


if __name__ == '__main__':
    main()


