"""Export path for a phone build. Does not report a latency or a file size.

PyTorch -> ONNX -> TensorFlow -> TFLite INT8. This machine has no phone and
this script does not write a fabricated size or latency.
"""
from __future__ import annotations

import sys


def main() -> None:
    print("PROTOCOL_READY")
    print("Target: TFLite INT8, size under 30 MB, CPU latency under 150 ms.")
    print("Those quantities are NOT_MEASURED. No phone was attached.")
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        print("tensorflow is not installed; conversion was not run.")
        raise SystemExit(0)
    print("tensorflow is importable. Conversion was not run in this entry point.")


if __name__ == "__main__":
    main()
