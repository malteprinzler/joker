from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity, _lpips_update
from torch import Tensor


class LPIPS(LearnedPerceptualImagePatchSimilarity):
    def __init__(self, *args, **kwargs):
        """
        defines lpips metric equivalent to torchmetrics.LearnedPerceptualImagePatchSimilarity but enables weighting of batches
        """
        super().__init__(*args, **kwargs)

    def update(self, img1: Tensor, img2: Tensor, weights: Tensor = None) -> None:
        """Update internal states with lpips score."""
        loss, total = _lpips_update(img1, img2, net=self.net, normalize=self.normalize)
        if weights is not None:
            loss = loss * weights
        self.sum_scores += loss.sum()
        self.total += total
