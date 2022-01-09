from typing import Tuple
from path import Path
import editdistance

from model import Model
from dataloader import DataLoader

# train pipe
def train_pipe(model: Model,
               loader: DataLoader,
               line_mode: bool,
               early_stopping: int=25):

    """training pipeline"""


# validate pipe
def validate_pipe(model: Model,
                  loader: DataLoader,
                  line_mode: bool) -> Tuple[float, float]:

    """Validate pipeline"""

# infer pipe
def infer_pipe(model: Model,
               fn_img: Path):

    """infer pipeline"""