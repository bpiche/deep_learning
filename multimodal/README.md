# DocumentAI for Research Papers

## Installation

Create a new python3.10 virtual environment

```
python3.10 -m venv ~/.venv/deep
source ~/.venv/deep/bin/activate
(deep) $> python -m pip install --upgrade pip
(deep) $> python -m pip install -r requirements.txt
```

You might need to install `transformers` from source for Nougat
```
(deep) $> pip install git+https://github.com/huggingface/transformers
```

Find a few research papers and dump them into the `./data` subdirectory
```
./data/
    - dit_lm.pdf
    - document_ai_retrospective.pdf
    - layoutlmv3.pdf
    - lilt_lm.pdf
```

All you gotta do now is run the mainfile with the right arguments
```
(deep) $> python pipeline.py --filename data/lilt_lm.pdf
```

It's gonna write a couple of files back to the `/.data` folder
```
./data/
    - lilt_lm.md # Nougat output
    - lilt_lm.json # my postprocessed output
```