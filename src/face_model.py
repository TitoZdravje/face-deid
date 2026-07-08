from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class EmbeddingResult:
    embedding: np.ndarray | None
    face_detected: bool
    reason: str


class FaceRecognitionModel:
    """
    Base wrapper interface.

    The rest of the project should call get_embedding_result(),
    not a library-specific model directly.
    """

    name = "base"

    def get_embedding_result(self, image: Image.Image) -> EmbeddingResult:
        raise NotImplementedError


class FaceNetPytorchModel(FaceRecognitionModel):
    """
    Face-recognition wrapper using:
        - MTCNN for face detection/alignment
        - InceptionResnetV1 for embeddings

    This is the first baseline model.
    """

    name = "facenet_pytorch_vggface2"

    def __init__(self, device: str | None = None):
        try:
            from facenet_pytorch import InceptionResnetV1, MTCNN
        except ImportError as exc:
            raise ImportError(
                "facenet-pytorch is not installed. Run: pip install facenet-pytorch"
            ) from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
        )

        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    @torch.no_grad()
    def get_embedding_result(self, image: Image.Image) -> EmbeddingResult:
        image = image.convert("RGB")

        face_tensor = self.mtcnn(image)

        if face_tensor is None:
            return EmbeddingResult(
                embedding=None,
                face_detected=False,
                reason="no_face_detected",
            )

        if face_tensor.ndim != 3:
            return EmbeddingResult(
                embedding=None,
                face_detected=False,
                reason=f"unexpected_face_tensor_shape_{tuple(face_tensor.shape)}",
            )

        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        embedding = self.model(face_tensor)
        embedding = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)

        return EmbeddingResult(
            embedding=embedding,
            face_detected=True,
            reason="ok",
        )
