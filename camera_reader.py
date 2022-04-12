from abc import ABCMeta, abstractmethod
from time import time

import cv2
import numpy as np


class NoCameraException(Exception):
    pass


class _CameraReader(object, metaclass=ABCMeta):
    def __init__(self, need_timestamps, mirror, delay):
        self.need_timestamps = need_timestamps
        self.mirror = mirror
        self.delay = delay
        if self.delay:
            self.buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        return self.read()

    def read(self):
        """
        :return:
            If created with need_timestamps = True, a tuple:
                (frame_gray, timestamp)
            Otherwise:
                frame_gray
        """
        if not self.delay:
            return self._read_immediate()
        else:
            while len(self.buffer) < self.delay:
                self.buffer.append(self._read_immediate())
            return self.buffer.pop(0)

    def _read_immediate(self):
        frame_gray, timestamp = self._read_basic()
        if self.mirror:
            cv2.flip(frame_gray, 1, dst=frame_gray)
        if self.need_timestamps:
            return frame_gray, timestamp
        else:
            return frame_gray

    @abstractmethod
    def _read_basic(self):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, type, value, tb):
        pass


class FlyCaptureReader(_CameraReader):
    """
    Install flycapture 2:
        pip install Cython
        pip install git+https://github.com/jordens/pyflycapture2.git (or from ./pyflycapture2-master.zip)
    """
    def __init__(self, need_timestamps, mirror, delay):
        super().__init__(need_timestamps, mirror, delay)
        try:
            import flycapture2 as fly
        except ImportError as e:
            raise NoCameraException() from e

        self.cam = fly.Context()

        numCams = self.cam.get_num_of_cameras()
        if numCams == 0:
            raise NoCameraException("No FlyCapture cameras")

        self.uid = self.cam.get_camera_from_index(0)
        self.is_capturing = False

    def _set_prop(self, prop, key, value):
        defs = self.cam.get_property(prop)
        defs[key] = value
        self.cam.set_property(**defs)

    def __enter__(self):
        import flycapture2 as fly
        self.cam.connect(*self.uid)
        # Set image size and pixel format
        fmt7imgSet = self.cam.set_format7_configuration(fly.MODE_0, 0, 152, 1280, 720, fly.PIXEL_FORMAT_MONO8)

        # Enable auto-exposure
        self._set_prop(fly.GAIN, "auto_manual_mode", True)
        self._set_prop(fly.SHUTTER, "auto_manual_mode", True)
        self._set_prop(fly.AUTO_EXPOSURE, "auto_manual_mode", True)
        self._set_prop(fly.FRAME_RATE, "abs_value", 30)

        # Start
        self.cam.start_capture()
        self.is_capturing = True
        return self

    def __exit__(self, type, value, tb):
        self.cam.stop_capture()
        self.cam.disconnect()
        self.is_capturing = False

    def _read_basic(self):
        import flycapture2 as fly
        if not self.is_capturing:
            raise RuntimeError("The reader needs to be entered first, using 'with'")
        timestamp = time()
        frame_gray = fly.Image()
        self.cam.retrieve_buffer(frame_gray)
        return np.array(frame_gray), timestamp


class OpenCVCameraReader(_CameraReader):
    def __init__(self, need_timestamps, mirror, delay):
        super().__init__(need_timestamps, mirror, delay)
        self.target_width = 1280
        self.target_height = 720
        self.cap = cv2.VideoCapture()
        self._try_open_camera()

    def _try_open_camera(self):
        self.cap.open(0)
        if not self.cap.isOpened():
            raise NoCameraException("OpenCV couldn't open the camera")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if actual_width != self.target_width or actual_height != self.target_height:
            raise RuntimeError(f"Couldn't set the frame size to "
                f"{self.target_width}x{self.target_height}: the reported size is "
                f"{actual_width}x{actual_height}")
        self.cap.set(cv2.CAP_PROP_MONOCHROME, 1)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)

    def __enter__(self):
        if not self.cap.isOpened():
            self._try_open_camera()
        return self

    def __exit__(self, type, value, tb):
        self.cap.release()

    def _read_basic(self):
        success, frame_bgr = self.cap.read()
        if not success:
            raise RuntimeError("Couldn't read a new frame")
        timestamp = time()
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return frame_gray, timestamp


def create_camera_reader(need_timestamps, mirror, delay=0):
    implementations = [FlyCaptureReader, OpenCVCameraReader]
    exceptions = []
    for impl in implementations:
        try:
            return impl(need_timestamps, mirror, delay)
        except NoCameraException as exc:
            exceptions.append(exc)
    raise NoCameraException(exceptions)
