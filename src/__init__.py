import json
from os import getcwd
from os.path import join

# root
root = getcwd()

# load config.json
config = json.load(join(root, "config/config.json"))

# files 
files = config.files

# decoder mapping
decoder_mapping = {
    "bestpath": config.decoder_type.best_path,
    "beamsearch": config.decoder_type.beam_search,
    "wordbeamsearch": config.decoder_type.beam_search
}