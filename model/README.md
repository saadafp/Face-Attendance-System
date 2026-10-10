# Models

This folder is intentionally empty. Model files are not included in the repository.

## Models used

This project uses two ONNX models from the **buffalo_m** pack of
[InsightFace](https://github.com/deepinsight/insightface):

| File | Purpose |
|------|---------|
| `det_2.5g.onnx` | Face detection + 5 landmarks (SCRFD) |
| `w600k_r50.onnx` | Face embedding (ArcFace, ResNet-50, 512-d) |

## Setup

Place both files here:

    models/
    └── buffalo_m/
        ├── det_2.5g.onnx
        └── w600k_r50.onnx

To download the pack, run once:

    python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_m')"

then copy the two files from `~/.insightface/models/buffalo_m/` into this folder.

## License

- InsightFace **source code**: MIT License.
- InsightFace **pretrained models** (including buffalo_m): provided for
  **non-commercial research purposes only**. See the License section of the
  official repository: https://github.com/deepinsight/insightface#license

This project is a portfolio/educational project.