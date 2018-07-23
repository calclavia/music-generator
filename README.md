# DeepJ: A model for style-specific music generation
https://arxiv.org/abs/1801.00887

## Abstract
Recent advances in deep neural networks have enabled algorithms to compose music that is comparable to music composed by humans. However, few algorithms allow the user to generate music with tunable parameters. The ability to tune properties of generated music will yield more practical benefits for aiding artists, filmmakers, and composers in their creative tasks. In this paper, we introduce DeepJ - an end-to-end generative model that is capable of composing music conditioned on a specific mixture of composer styles. Our innovations include methods to learn musical style and music dynamics. We use our model to demonstrate a simple technique for controlling the style of generated music as a proof of concept. Evaluation of our model using human raters shows that we have improved over the Biaxial LSTM approach.


## Check Google colab notebook DeepJ_Main for dependencies fixing

## Requirements
- Python 3.5 (Not 3.6!!!!!)
- Pytorch 4.0 (http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl)
- Cuda 9.1 (8.0 for Colab)
- python3-midi (Check DeepJ_Main notebook for installation)


```
Run following in DeepJ directory
pip install -r requirements.txt
```

The dataset is not provided in this repository. To train a custom model, you will need to include a MIDI dataset in the `data/` folder in following ways: data/classical, data/romantic, and each such folder contains midi file of corresponding .mid files.
