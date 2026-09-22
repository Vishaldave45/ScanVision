from dataclasses import dataclass, field


@dataclass
class OCRWord:
    """Represents a single recognized word with geometry and confidence."""
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


@dataclass
class OCRResult:
    """Encapsulates text recognition outputs and word-level metadata."""
    text: str
    words: list[OCRWord] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def average_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)
