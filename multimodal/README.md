# DocumentAI for Research Papers

## Installation

Create a new python3.10 virtual environment

```
python3.10 -m venv ~/.venv/deep
source ~/.venv/deep/bin/activate
(deep) $> python -m pip install --upgrade pip
```

Install `transformers` from source (for Nougat)
```
(deep) $> pip install git+https://github.com/huggingface/transformers
```

Install the rest of the dependencies. The main ones are
`Pillow torch pymupdf markdown-to-json`
```
(deep) $> python -m pip install -r requirements.txt
```
You probably need to install the spaCy model separately too
```
(deep) $> python -m spacy download en_core_web_lg
```

### DocumentAI

Find a few research papers and dump them into the `./data` subdirectory. You can use any papers you like.
```
./data/
    - dit_lm.pdf
    - document_ai_retrospective.pdf
    - layoutlmv3.pdf
    - lilt_lm.pdf
```

All you gotta do now is run the mainfile with the right arguments. If you use the `--all` flag, it will read all of the pdfs in that directory, one at a time. You can alternatively use the `--filename` flag if you just want to point it to one pdf.
```
(deep) $> python pipeline.py [--all|--filename]
```

It's gonna write a couple of files back to the `/.data` folder. The Nougat output is a markdown file with all of the fields from the original paper. I take those files and apply a bunch of string replacement rules on them to normalize the markdown headers and the spelling of the main fields (Title, Authors, Abstract, Key Findings, References). 
```
./data/
    ...
    - lilt_lm.md # Nougat output
    - lilt_lm.json # my postprocessed output
    ...
```

### Search

Once you have written a few .json responses to the `/.data` subdirectory, you will be able to effectively use the search function. Make sure you give it a `--query` or else it will default to using the 'Show me a paper about the LayoutLMv3 model' query. This function quietly writes a `response.json` file to the `./data` subdirectory. It's a ranked list of (Title, Author, Abstract, Key Findingss, similarity) dicts, sorted by (cosine) similiarity score.
```
(deep) $> python search.py --query "Show me the DIT LM model paper"
```

### Visual QA

I was interested in the Donut model and its ability to perform visual question answering, so here you go. This one example actually works fairly well, but almmost nothing else I tried worked. Don't say I didn't warn you!
```
(deep) $> python visual_qa.py --filename lilt_lm.pdf --prompt "What is the title of this paper?"
```