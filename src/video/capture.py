
from __future__ import annotations

import time
from typing import Union, Tuple

import cv2
from loguru import logger


class VideoReader:
    """
    Lector simple para webcam/RTSP/archivo.
    - Si se cae RTSP, intenta reconectar.
    """

    def __init__(
        self,
        source: Union[int, str],
        frame_size: Tuple[int, int] = (1280, 720),
        reconnect: bool = True,
        reconnect_wait_s: float = 1.0,
    ) -> None:
        self.source = source
        self.frame_w, self.frame_h = frame_size
        self.reconnect = reconnect
        self.reconnect_wait_s = reconnect_wait_s
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        logger.info(f"Abriendo video source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)

        if isinstance(self.source, int):
            # Solo aplica a webcam normalmente
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)

        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {self.source}")

    def read(self):
        if self.cap is None:
            self.open()

        assert self.cap is not None
        ok, frame = self.cap.read()

        if ok and frame is not None:
            return True, frame

        # Si falla lectura (especial RTSP), intentamos reconectar
        if self.reconnect:
            logger.warning("Frame falló. Reintentando reconexión...")
            self.release()
            time.sleep(self.reconnect_wait_s)
            try:
                self.open()
            except Exception as e:
                logger.error(f"Reconexión falló: {e}")
                return False, None
            return self.cap.read()

        return False, None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
