from .optical_scanner import OpticalScanner

try:
    from .cv_scanner import CVScanner
except ImportError:
    CVScanner = None  # PyTorch not installed — CV scanner unavailable

__all__ = ["OpticalScanner", "CVScanner"]
