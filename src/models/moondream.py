from typing import Union, Iterator, List
from models.fast_vlm import BaseFastVLM, ImgLike


class Moondream_FastVLM(BaseFastVLM):
    """Minimal wrapper for Moondream2 with streaming support."""

    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        revision: str = "2025-06-21",
        device_map: Union[str, dict] = "auto",
    ):
        """
        Initialize the Moondream2 model wrapper.

        Args:
            model_id (str): Hugging Face model ID to load.
            revision (str): Model revision or commit hash.
            device_map (Union[str, dict]): Device mapping strategy for model loading.
        """
        super().__init__(model_id, revision, device_map)

    def caption(self, img: ImgLike, length: str = "normal") -> str:
        """
        Generate a full image caption.

        Args:
            img (ImgLike): Path, URL, or PIL image to caption.
            length (str): Caption length. One of {"short", "normal"}.

        Returns:
            str: Caption text for the input image.
        """
        return self._run_text_task("caption", img, length=length)

    def stream_caption(self, img: ImgLike, length: str = "normal") -> Iterator[str]:
        """
        Stream caption tokens incrementally.

        Args:
            img (ImgLike): Path, URL, or PIL image to caption.
            length (str): Caption length. One of {"short", "normal"}.

        Yields:
            Iterator[str]: Caption tokens as they are generated.
        """
        return self._run_text_task("caption", img, length=length, stream=True)

    def query(self, img: ImgLike, question: str) -> str:
        """
        Answer a question about an image (VQA).

        Args:
            img (ImgLike): Path, URL, or PIL image to analyze.
            question (str): Natural language question about the image.

        Returns:
            str: Answer to the question.
        """
        return self._run_text_task("query", img, question)

    def stream_query(self, img: ImgLike, question: str) -> Iterator[str]:
        """
        Stream VQA answer tokens incrementally.

        Args:
            img (ImgLike): Path, URL, or PIL image to analyze.
            question (str): Natural language question about the image.

        Yields:
            Iterator[str]: Answer tokens as they are generated.
        """
        return self._run_text_task("query", img, question, stream=True)

    def detect(self, img: ImgLike, label: str) -> List[dict]:
        """
        Detect objects of a given label in an image.

        Args:
            img (ImgLike): Path, URL, or PIL image to analyze.
            label (str): Object label to detect.

        Returns:
            List[dict]: List of detected objects with bounding box coordinates.
        """
        return self.model.detect(self._to_image(img), label)["objects"]

    def point(self, img: ImgLike, label: str) -> List[dict]:
        """
        Locate points for a given label in an image.

        Args:
            img (ImgLike): Path, URL, or PIL image to analyze.
            label (str): Label for which to locate points.

        Returns:
            List[dict]: List of points with (x, y) coordinates.
        """
        return self.model.point(self._to_image(img), label)["points"]
