# JLU Buff TensorRT model directory

Place the exported JLU-style buff detector model here:

- `buff.onnx`: ONNX model used to build a TensorRT 10.3 engine automatically.
- `buff_trt10_3.engine`: serialized TensorRT engine loaded directly when present.

The detector adapter expects each candidate to expose bbox + confidence + five keypoints in JLU order:
`r_center`, `bottom_right`, `top_right`, `top_left`, `bottom_left`.
