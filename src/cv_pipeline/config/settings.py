from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectionConfig:
    canny_low: int = 50
    canny_high: int = 150
    edge_blur_kernel: tuple[int, int] = (5, 5)
    thresh_blur_kernel: tuple[int, int] = (9, 9)
    min_area_ratio: float = 0.15
    morph_kernel_size: tuple[int, int] = (11, 11)


@dataclass
class OCRPreprocessConfig:
    adaptive_block_size: int = 11
    adaptive_constant: int = 2
    opening_kernel_size: tuple[int, int] = (3, 3)
    closing_kernel_size: tuple[int, int] = (3, 3)


@dataclass
class OCRConfig:
    psm: int = 3
    oem: int = 3
    lang: str = "eng"
    use_preprocessed: bool = False  # If False, run OCR on rectified original; if True, on cleaned image


@dataclass
class PipelineConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr_preprocess: OCRPreprocessConfig = field(default_factory=OCRPreprocessConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
