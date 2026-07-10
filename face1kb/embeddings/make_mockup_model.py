"""Generate a mockup ONNX model that mimics the proprietary-model interface.

The three proprietary Innovatrics face-recognition models evaluated in the
paper (``inno-fast``, ``inno-balanced``, ``inno-accurate``) are commercial
products and are **not distributed** with this repository. To keep the
pipeline runnable end to end, this script builds a small deterministic
network with an identical interface:

- input tensor ``input.1`` of shape ``(1, 3, 112, 112)``, float32, holding a
  BGR face crop normalised to ``[-1, 1]``,
- output tensor of shape ``(1, 512)``, the L2-normalised face embedding.

The graph is 8x8 average pooling followed by a fixed random projection
(seed 42) and L2 normalisation - the embeddings carry **no biometric
meaning** and metrics computed on them are placeholders. To reproduce the
paper's proprietary-model results, obtain the models from Innovatrics and
place them under ``models/proprietary/`` (see README).

Usage
-----
    python -m face1kb.embeddings.make_mockup_model
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from face1kb import config

POOLED_FEATURES = 3 * 14 * 14  # 8x8 average pooling of a 3x112x112 input
EMBEDDING_DIM = 512


def build_model() -> onnx.ModelProto:
    """Assemble the deterministic mockup graph."""
    rng = np.random.default_rng(seed=42)
    projection = rng.normal(
        scale=1.0 / np.sqrt(POOLED_FEATURES),
        size=(POOLED_FEATURES, EMBEDDING_DIM),
    ).astype(np.float32)

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "AveragePool",
                ["input.1"],
                ["pooled"],
                kernel_shape=[8, 8],
                strides=[8, 8],
            ),
            helper.make_node("Reshape", ["pooled", "flat_shape"], ["flat"]),
            helper.make_node("MatMul", ["flat", "projection"], ["projected"]),
            helper.make_node(
                "ReduceL2", ["projected"], ["norm"], axes=[1], keepdims=1
            ),
            helper.make_node("Div", ["projected", "norm"], ["embedding"]),
        ],
        name="mockup_embedder",
        inputs=[
            helper.make_tensor_value_info(
                "input.1", TensorProto.FLOAT, [1, 3, 112, 112]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "embedding", TensorProto.FLOAT, [1, EMBEDDING_DIM]
            )
        ],
        initializer=[
            numpy_helper.from_array(projection, name="projection"),
            numpy_helper.from_array(
                np.array([1, POOLED_FEATURES], dtype=np.int64),
                name="flat_shape",
            ),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="face1kb-mockup",
    )
    # Pin a conservative IR version so any onnxruntime release loads it.
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def export(path=None) -> None:
    """Write the mockup model to *path* (default from config)."""
    path = path or config.MOCKUP_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), str(path))
    print(f"Exported mockup model to {path}")


if __name__ == "__main__":
    export()
