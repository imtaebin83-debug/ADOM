# OpenCV headless metadata shim

Albumentations 1.3.1 and qudida declare `opencv-python-headless` as a
distribution dependency. ADOM deliberately installs `opencv-python==4.8.0.76`
as the sole provider of the shared `cv2` namespace.

This package contains no Python modules or native libraries. It only satisfies
that distribution-name dependency so the final `pip check` can remain strict
without installing two wheels that overwrite the same `cv2` files. Docker's
final sanity check asserts both the real `opencv-python` distribution and
`cv2.__version__`.
