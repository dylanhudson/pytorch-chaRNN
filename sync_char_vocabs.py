#this script finds the char level vocab differences between the files 
# listed as command line arguments, and adds chars to each file such that
#all have the same set of chars. this allows them to have vocab compatibility 
# when they're made into  datasets by textdataset.py for training the nn model


import sys

def get_char_vocab(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return set(text)

def add_missing_chars_to_file(filename, missing_chars):
    with open(filename, 'a', encoding='utf-8') as f:
        for char in missing_chars:
            f.write(char)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 sync_char_vocabs.py <file1> <file2> ... <fileN>")
        sys.exit(1)

    file_list = sys.argv[1:]
    all_chars = set()

    # Collect all unique characters from all files
    for filename in file_list:
        file_chars = get_char_vocab(filename)
        all_chars.update(file_chars)

    # For each file, find missing characters and append them
    for filename in file_list:
        file_chars = get_char_vocab(filename)
        missing_chars = all_chars - file_chars
        if missing_chars:
            add_missing_chars_to_file(filename, missing_chars)
            print(f"Added {len(missing_chars)} missing chars to {filename}")
        else:
            print(f"No missing chars for {filename}")


