#!/usr/bin/env python3
"""
Generate test fixtures for onnxcc:
- A 4 -> 8 -> 2 MLP model at ONNX opset 13 using MatMul, Add, Relu nodes.
- A (1, 4) float32 raw binary input file (16 bytes).

Node count: 6 (2 x MatMul, 2 x Add, 2 x Relu)
Initializer count: 4 (W1, B1, W2, B2)
"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto


def main():
    # Set fixed seed for reproducibility
    np.random.seed(42)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(output_dir, exist_ok=True)

    # Weights and Biases initialization
    W1_data = np.random.randn(4, 8).astype(np.float32)
    B1_data = np.random.randn(1, 8).astype(np.float32)
    W2_data = np.random.randn(8, 2).astype(np.float32)
    B2_data = np.random.randn(1, 2).astype(np.float32)

    # Initializers
    W1_init = helper.make_tensor("W1", TensorProto.FLOAT, [4, 8], W1_data.flatten().tolist())
    B1_init = helper.make_tensor("B1", TensorProto.FLOAT, [1, 8], B1_data.flatten().tolist())
    W2_init = helper.make_tensor("W2", TensorProto.FLOAT, [8, 2], W2_data.flatten().tolist())
    B2_init = helper.make_tensor("B2", TensorProto.FLOAT, [1, 2], B2_data.flatten().tolist())

    # Inputs and Outputs
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

    # Nodes (Total: 6 nodes)
    node1 = helper.make_node("MatMul", ["input", "W1"], ["matmul1_out"])
    node2 = helper.make_node("Add", ["matmul1_out", "B1"], ["add1_out"])
    node3 = helper.make_node("Relu", ["add1_out"], ["relu1_out"])
    node4 = helper.make_node("MatMul", ["relu1_out", "W2"], ["matmul2_out"])
    node5 = helper.make_node("Add", ["matmul2_out", "B2"], ["add2_out"])
    node6 = helper.make_node("Relu", ["add2_out"], ["output"])

    nodes = [node1, node2, node3, node4, node5, node6]
    initializers = [W1_init, B1_init, W2_init, B2_init]

    # Graph
    graph_def = helper.make_graph(
        nodes=nodes,
        name="MLP_4_8_2",
        inputs=[input_info],
        outputs=[output_info],
        initializer=initializers,
    )

    # Model (opset 13)
    opset_imports = [helper.make_opsetid("", 13)]
    model_def = helper.make_model(graph_def, producer_name="onnxcc-fixture", opset_imports=opset_imports)

    # Validate model with onnx.checker
    onnx.checker.check_model(model_def)

    # Save model
    model_path = os.path.join(output_dir, "mlp.onnx")
    onnx.save(model_def, model_path)

    # Generate (1, 4) float32 raw binary input (16 bytes)
    input_data = np.random.randn(1, 4).astype(np.float32)
    input_bin_path = os.path.join(output_dir, "input_1x4.bin")
    input_data.tofile(input_bin_path)

    # Print summary & verify op types
    op_types = [node.op_type for node in model_def.graph.node]
    print(f"Generated ONNX model: {model_path}")
    print(f"Node count: {len(nodes)}")
    print(f"Initializer count: {len(initializers)}")
    print(f"Op types in graph: {op_types}")
    print(f"Generated input binary: {input_bin_path} ({os.path.getsize(input_bin_path)} bytes)")


if __name__ == "__main__":
    main()
