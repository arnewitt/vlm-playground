from pathlib import Path
from typing import Literal, Union
from PIL import Image
from transformers import AutoModelForCausalLM

ImgLike = Union[str, Path, Image.Image]


class BaseFastVLM:
    """Base wrapper for Fast VLM models."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        device_map: Union[str, dict] = "auto",
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            device_map=device_map,
        )

    @staticmethod
    def _to_image(img: ImgLike) -> Image.Image:
        """Accept path/str/PIL and return a RGB PIL.Image."""
        return (
            img if isinstance(img, Image.Image) else Image.open(str(img)).convert("RGB")
        )

    def _run_text_task(
        self,
        task: Literal["caption", "query"],
        img: ImgLike,
        *args,
        stream: bool = False,
        **kwargs,
    ):
        """Internal text task runner with optional streaming."""
        image = self._to_image(img)
        method = getattr(self.model, task)
        key = "caption" if task == "caption" else "answer"
        if stream:
            return method(image, *args, stream=True, **kwargs)[key]
        return method(image, *args, **kwargs)[key]
