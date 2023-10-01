import re
import markdown_to_json

filename = "data/lilt_lm.md"
# read all the lines in the file
file = open(filename, 'r')
lines = file.readlines()
# add '# Title\n' to the first line
lines[0] = '# Title\n' + lines[0]
# replace multiple # characters with a single # character
lines = [re.sub(r'#+', '#', line) for line in lines]

dictified = markdown_to_json.dictify("".join(lines))
# return the value of the 'Abstract' key
print(dictified.keys())