from typing import Tuple
import numpy as np
import logging
import random
import cv2
import pickle
from path import Path
from collections import namedtuple
import lmdb

Sample = namedtuple("Sample", "gt_text, file_path")
Batch = namedtuple("Batch", "imgs, gt_texts, batch_size")

class DataLoader():

    """
    To load data which corresponds to IAM format
    """

    def __init__(self,
                 data_dir: Path,
                 batch_size: int,
                 data_split: float=0.5,
                 fast: bool=True):

        if data_dir.exists():
            logging.info("data directory exists!")

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []
        self.fast = fast
        if fast:
            self.env = lmdb.open(data_dir + "/lmdb", readonly=True)
        
        file = open(data_dir + "/gt/words.txt")
        chars = set()
        bad_samples = ["r06-022-03-05", "a01-117-05-02"]
        for i in range(len(file)):

            if not file[i] or file[i][0] == "#":
                continue

            line_split = file[i].strip().split(' ')

            file_name_split = line_split[0].split("-")
            file_name_subdir1 = file_name_split[0]
            file_name_subdir2 = "{0}-{1}" % (file_name_split[0], file_name_split[1])
            file_basename = line_split[0] + ".png"
            file_name = "".join([
                data_dir, "/img", "/", file_name_subdir1, "/", file_name_subdir2, "/", file_basename
            ])

            if line_split[0] in bad_samples:
                logging.info("Ignore the broken images: ", file_name)
                continue

            # GT text are columns starting at 9
            gt_text = " ".join(line_split[8:])
            chars = chars.union(set(list(gt_text)))

            # put sample into list
            self.samples.append(Sample(gt_text, file_name))

        # train and validation split
        split_idx = int(data_split * len(self.samples))
        self.train_samples, self.validation_samples = self.samples[:split_idx], self.samples[split_idx:]

        # put words into lists
        self.train_words = [i.gt_text for i in self.train_samples]
        self.validation_words = [i.gt_text for i in self.validation_samples]

        # start with train set
        self.train_set()

        # list of all chars in dataset
        self.char_list = sorted(list(chars))

    def train_set(self):

        """Switch to randomly chosen subset of training dataset."""

        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = "train"

    def validation_set(self):

        """Switch to validation dataset"""

        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = "val"

    def get_iterator_info(self) -> Tuple[int, int]:

        """Current batch index and overall number of batches"""

        if self.curr_set == "train":
            num_batches = int(np.floor(
                len(self.samples) / self.batch_size
            ))
        else:
            num_batches = int(np.ceil(
                len(self.samples) / self.batch_size
            ))
        curr_batch = self.curr_idx // self.batch_size + 1

        return curr_batch, num_batches
    
    def has_next(self) -> bool:

        """any next item?"""

        if self.curr_set == "train":
            return self.curr_idx + self.batch_size <= len(self.samples)
        else:
            return self.curr_idx < len(self.samples)

    def _get_img(self, index:int) -> np.ndarray:

        if self.fast:
            with self.env.begin() as txn:
                basename = Path(self.samples[index].file_path).basename()
                data = txn.get(basename.encode("ascii"))
                img = pickle.loads(data)
        else:
            img = cv2.imread(self.samples[index].file_path, cv2.IMREAD_GRAYSCALE)

        return img

    def get_next(self) -> Batch:

        """Get next item"""

        imgs = [self._get_img(i) for i in range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))]
        gt_texts = [self.samples[i].gt_text for i in range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))]
        self.curr_idx += self.batch_size

        return Batch(imgs, gt_texts, len(imgs))