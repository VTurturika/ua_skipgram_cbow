import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('file', metavar='file', type=str)
args = parser.parse_args()

with open(args.file, 'r') as fin:
	with open('{0}_prepared'.format(args.file), 'w') as fout:
		for line in fin:
			processed_line = line.strip().lower()
			processed_line = re.sub('\\s\\d+[а-яєіїґ-]*\\s', ' ', processed_line)
			processed_line = re.sub('[а-яєіїґ-]\\.\\s?', ' ', processed_line)
			processed_line = re.sub('[^а-яєіїґ ]', '', processed_line)
			processed_line = processed_line.strip()
			if processed_line:
				fout.write(processed_line + '\n')
    
        
            
