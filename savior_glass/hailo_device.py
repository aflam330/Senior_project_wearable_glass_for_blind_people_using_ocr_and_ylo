"""One shared handle on the Raspberry Pi AI HAT (Hailo-8) for every SARA model that runs on it.

The chip accepts one open device per process (a second VDevice() fails with HAILO_OUT_OF_PHYSICAL_DEVICES), so the
object detector, the text detector and the face detector all configure their model on this one device. HailoRT's
model scheduler (round robin) switches the chip between them; a model is never activated by hand.

open_model(hef_path) -> HailoModel with infer(batch_uint8) -> {output name: float32 array}; close() releases it, and
the device is released when the last model closes.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_vdevice = None
_users = 0


def _acquire():
    global _vdevice, _users
    from hailo_platform import HailoSchedulingAlgorithm, VDevice
    with _lock:
        if _vdevice is None:
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            _vdevice = VDevice(params)
        _users += 1
        return _vdevice


def _release():
    global _vdevice, _users
    with _lock:
        _users = max(0, _users - 1)
        if _users == 0 and _vdevice is not None:
            try:
                _vdevice.release()
            finally:
                _vdevice = None


class HailoModel:
    def __init__(self, hef_path: str) -> None:
        from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
                                    InputVStreamParams, OutputVStreamParams)
        self._pipe = None
        device = _acquire()
        try:
            hef = HEF(hef_path)
            self.inputs = hef.get_input_vstream_infos()
            self.outputs = hef.get_output_vstream_infos()
            self.group = device.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
            self._ctx = InferVStreams(self.group, InputVStreamParams.make(self.group, format_type=FormatType.UINT8),
                                      OutputVStreamParams.make(self.group, format_type=FormatType.FLOAT32))
            self._pipe = self._ctx.__enter__()
        except Exception:
            _release()
            raise
        self._infer_lock = threading.Lock()

    def infer(self, batch) -> dict:
        import numpy as np
        with self._infer_lock:
            return self._pipe.infer({self.inputs[0].name: np.ascontiguousarray(batch)})

    def close(self) -> None:
        if self._pipe is None:
            return
        try:
            self._ctx.__exit__(None, None, None)
        finally:
            self._pipe = None
            _release()


def open_model(hef_path: str) -> HailoModel:
    return HailoModel(hef_path)
