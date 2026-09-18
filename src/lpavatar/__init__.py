# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0

"""lpavatar -- license-clean avatar registration / rendering building blocks.

This package is an independent implementation of a LivePortrait-style
talking-head pipeline, written from the upstream open-source references only:

- LivePortrait (MIT) -- crop geometry, implicit-keypoint transform,
  stitching / warping / SPADE decoder graphs
- ditto-talkinghead (Apache-2.0) -- portable ONNX graphs and numpy
  reference wrappers
- MediaPipe (Apache-2.0) -- BlazeFace detector, Face Mesh landmarks
- MODNet (Apache-2.0) -- portrait matting

Layout::

    lpavatar.runtime       ONNX Runtime engine wrapper (CPU / CUDA)
    lpavatar.registration  portrait -> Avatar (crop, landmarks, f_s, kp_info)
    lpavatar.render        (x_s, x_d) -> 512 px face via stitch / warp / decoder
    lpavatar.compose       paste-back, matting, pixel-format conversion
    lpavatar.artifacts     download of the public ONNX graphs

    lpavatar.motion        ditto LMDM audio -> motion adapter (streaming)

The audio -> motion stage lives in ``lpavatar.motion`` and is a port of the
Apache-2.0 ditto-talkinghead online pipeline; see docs/DITTO_AVATAR_HANDOFF.md.
"""

from lpavatar.registration.avatar import Avatar, AvatarRegistrar, RegistrationConfig
from lpavatar.render.keypoints import KPInfo

__all__ = ["Avatar", "AvatarRegistrar", "KPInfo", "RegistrationConfig"]
