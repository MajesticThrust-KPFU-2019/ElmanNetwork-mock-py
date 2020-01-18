import sys
from typing import TextIO, Dict


def get_characters(f: TextIO) -> Dict[str, int]:
    text = f.read()
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1

    return freq


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("This program lists all characters used in the dataset")
        print("Specify a txt dataset")
        sys.exit()
    else:
        with open(sys.argv[1]) as f:
            chars = get_characters(f)

        
        kv = list(chars.items())
        kv.sort(key=lambda pair: pair[1], reverse=True)

        print("Total unique characters: {}".format(len(kv)))
        print("\nFrequencies:")
        for pair in kv:
            print("'{}'\t{}".format(pair[0], pair[1]))
        
        print("\nPython list")
        print(sorted(chars.keys()))
