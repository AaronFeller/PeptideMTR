# PeptideCLM
This work was developed in the [Wilke lab](https://wilkelab.org/) by Aaron Feller ([Department of Molecular Biosciences](https://molecularbiosci.utexas.edu/) at [The University of Texas at Austin](https://www.utexas.edu/)).

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Models](#models)
- [Tokenizer](#tokenizer)
- [Datasets](#datasets)
- [Contributing](#contributing)
- [License](#license)

## Introduction
PeptideMTR is a multi-task regression trained language model for peptide encoding.

## Getting Started
### Installation
To install the required libraries (all necessary dependencies not listed), run:
```
pip install torch transformers
```

### Usage
To use the pre-trained models, you can load them as follows:

```
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('aaronfeller/model_name')
tokenizer = 
```
Replace `'model_name'` with the desired model from the table below.


## Models

| Model name              | Training dataset                                          | Description                                                                                                               |
|-----------------------------|--------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| | |

All models hosted on huggingface can be loaded from [huggingface.co/aaronfeller](https://huggingface.co/aaronfeller).


## Tokenizer


## Datasets


## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License
The author(s) are protected under the MIT License - see the LICENSE file for details.

