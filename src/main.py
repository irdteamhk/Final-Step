from typing import Tuple, List
import argparse
import json
from path import Path
import cv2

from src import files, decoder_mapping
from lib.dataloader import DataLoader
from lib.model import *
from lib.pipe import *


# main function
def main():

    """main function to do handwriting OCR"""

    # arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", help="dataset directory.", type=Path, required=False)
    parser.add_argument("--mode", choices=["train", "validate", "inference"], default="inference")
    parser.add_argument("--decoder", choices=["bestpath", "beamsearch", "wordbeamsearch"], default="bestpath")
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=100)
    parser.add_argument("--line_mode", help="Train to read text lines instead of single words. ", action="store_true")
    parser.add_argument("--fast", help="Load samples from LMDB.", action="store_true")
    parser.add_argument("--img_file", help="Image used for inference.", type=Path, default="data/word.png")
    parser.add_argument("--early_stopping", help="Early stopping epochs.", type=int, default=50)
    parser.add_argument("--dump", help="Dump the output of NN to csv file(s).", action="store_true")
    args = parser.parse_args()

    # set chosen CTC decoder
    decoder_type = decoder_mapping[args.decoder]

    if args.mode in ("train", "validate"):

        # load training data and create model
        loader = DataLoader(args.data_dir, args.batch_size, fast=args.fast)
        char_list = loader.char_list

        # if in line mode, handle all whitespace
        if args.line_mode and " " not in char_list:
            char_list = [" "] + char_list

        # save characters of model for inference mode
        open(files.char_list, "w").write("".join(char_list))

        # save words contained in dataset into file
        open(files.corpus, "w").write(" ".join(loader.train_words + loader.validation_words))

        # train or validate
        if args.mode == "train":
            model = Model(char_list, decoder_type)
            train_pipe(
                model,
                loader,
                line_mode=args.line_mode,
                early_stopping=args.early_stopping
            )
        elif args.mode == "validate":
            model = Model(char_list, decoder_type, must_restore=True)
            validate_pipe(
                model,
                loader,
                args.line_mode
            )
    
    elif args.mode == "inference":
        model = Model(
            list(open(files.char_list).read()),
            decoder_type,
            must_restore=True,
            dump=args.dump
        )
        infer_pipe(model, args.img_file)
    
    return "finished handwriting OCR"


if __name__ == "__main__":
    main()