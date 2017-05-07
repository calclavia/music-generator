# DeepJ: A model for style-specific music generation
Recent advances in deep neural networks has enabled algorithms to compose music that is comparable to music composed by humans. However, few algorithms allow the user to generate music with tunable parameters. The ability to tune properties of generated music will yield more practical benefits for aiding artists, filmmakers and composers in their creative tasks. Our goal is to build an end-to-end generative model that is capable of composing music conditioned with a specific mixture of musical style as a proof of concept.

## Requirements
- Python 3.5

Clone Python MIDI (https://github.com/vishnubob/python-midi) and install the
Python3 branch using `python3 install setup.py`.

Install CUDA. See Tensorflow docs for more information.

Then, install other dependencies of this project.
```
pip install -r requirements.txt
```
