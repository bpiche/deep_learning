import re
import glob
import json
import argparse
import document_ai
import markdown_to_json


parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="data/lilt_lm.pdf")
parser.add_argument('--all', dest='all', action='store_true')
args = parser.parse_args()


def extract_fields(lines):
    """
    """
    # remove leading newlines
    lines = [line.lstrip() for line in lines]
    # remove any # characters from the beginning of the first line
    lines[0] = re.sub(r'^#+', '', lines[0])
    # append "# Authors" after the first new line in the first line
    lines[0] = re.sub(r'\n', '\n# Authors\n', lines[0], count=1)
    lines[0] = '# Title\n' + lines[0]
    # replace multiple # characters with a single # character
    lines = [re.sub(r'#+', '#', line) for line in lines]
    # create a dictionary and clean it up
    dictified = markdown_to_json.dictify("".join(lines))
    dictified = {re.sub(r'\W+', '', k): v for k, v in dictified.items()}
    dictified = {re.sub(r'\d+', '', k): v for k, v in dictified.items()}
    # convert camel case to individual words
    dictified = {re.sub(r'([a-z])([A-Z])', r'\1 \2', k): v for k, v in dictified.items()}
    dictified = {re.sub(r'Conclusion.*', 'Conclusion', k): v for k, v in dictified.items()}
    # cut everything in the abstract off after the first newline
    dictified['Abstract'] = dictified['Abstract'].split('\n')[0]
    # cut everything in the conclusion off after the first newline
    dictified['Conclusion'] = dictified['Conclusion'].split('\n')[0]
    # remove the `\\({}^{1}\\)` pattern from the authors value
    dictified['Authors'] = re.sub(r'\\({}^{.*}\\)', '', dictified['Authors'])
    response = {
        "Title": dictified['Title'],
        "Authors": dictified['Authors'],
        "Abstract": dictified['Abstract'],
        "Key Findings": dictified['Conclusion'],
        "References": dictified['References']
    }
    return response


if __name__ == "__main__":
    """
    python multimodal/document_ai.py
    """
    if args.all:
        # for all files in glob.glob('data/*.pdf'):
        for filename in glob.glob('data/*.pdf'):
            lines = document_ai.nougat(filename)
            response = extract_fields(lines)
            # write the response to a json file, pretty-printed
            with open(filename.replace(".pdf", ".json"), "w") as f:
                f.write(json.dumps(response, indent=4))
    else:
        # ocr the pdf and generate markdown
        filename = args.filename
        lines = document_ai.nougat(filename)
        response = extract_fields(lines)
        # write the response to a json file, pretty-printed
        with open(filename.replace(".pdf", ".json"), "w") as f:
            f.write(json.dumps(response, indent=4))
        