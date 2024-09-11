import argparse
import json
import sys
# sys.path.append('D:/work/Charsiu evaluation/g2p')
from g2p import infer
import os


def process_text(text, language, output_file=None):
    results = infer(text, language)
    if output_file:
        write_results_to_file(results, output_file)
    else:
        for result in results:
            print(json.dumps(result))


def process_file(file_path, language, output_file=None):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        process_text(text, language, output_file)


def process_directory(directory_path, language, output_file=None):
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {file_path}...")
            process_file(file_path, language, output_file)


def write_results_to_file(results, output_file):
    with open(output_file, 'a', encoding='utf-8') as file:
        for result in results:
            file.write(json.dumps(result) + '\n')


def main():
    parser = argparse.ArgumentParser(description='G2P Command Line Tool')
    parser.add_argument('--text', '-t', type=str, help='Direct input text for G2P processing')
    parser.add_argument('--file', '-f', type=str, help='Path to a text file for G2P processing')
    parser.add_argument('--dir', '-d', type=str, help='Path to a directory containing .txt files for G2P processing')
    parser.add_argument('--lang', '-l', type=str, default=None, help='Language for G2P processing')
    parser.add_argument('--output', '-o', type=str, help='Optional: Output file path for the results in JSON Lines format')

    args = parser.parse_args()

    if args.output:
        # Ensure the output file is empty before writing new results
        open(args.output, 'w').close()

    if args.text:
        process_text(args.text, args.lang, args.output)
    elif args.file:
        process_file(args.file, args.lang, args.output)
    elif args.dir:
        process_directory(args.dir, args.lang, args.output)
    else:
        print("No input provided. Use --text (-t), --file (-f), or --dir (-d) to input text for processing.")


if __name__ == '__main__':
    main()
