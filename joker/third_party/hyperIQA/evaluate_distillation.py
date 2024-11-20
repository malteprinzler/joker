import numpy as np

from process_celebvtext import score_img

import glob
from pathlib import Path

root = Path("")
pred_img_files = [Path(p) for p in sorted(glob.glob(str(root / "*-pred_*.png")))]

scores = [score_img(p) for p in pred_img_files]
print(np.mean(scores))
