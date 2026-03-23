"""Visual encoder: image → SDR → DAF currents."""

from snks.encoder.encoder import VisualEncoder
from snks.encoder.gabor import GaborBank
from snks.encoder.sdr import kwta, sdr_overlap, batch_overlap_matrix

__all__ = ["VisualEncoder", "GaborBank", "kwta", "sdr_overlap", "batch_overlap_matrix"]
