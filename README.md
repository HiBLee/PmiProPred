# PmiProPred:a novel method towards plant miRNA promoter prediction based on CNN-Transformer network and convolutional block attention mechanism
## Dataset
### training dataset:
* datasets/training_dataset.txt
### independent testing dataset:
* datasets/test1_23146.txt
* datasets/test2_37111.txt
## Usage
* DataEmbedding.py: this file is used to load the raw sequences, and then split each sequence into a set of tokens.
* Metrics.py: this file is used to calculate various metrics.
* Transformer.py: this file is used to construct the Transformer model.
* ModelConstruction.py: this file is used to construct the PmiProPred model, including multi-stream deep feature extraction module, multi-stream deep feature refinement module, and multi-layer perceptron module.
## Citation
Li HB, Meng J, Wang ZW, Luan YS. PmiProPred:a novel method towards plant miRNA promoter prediction based on CNN-Transformer network and convolutional block attention mechanism. (under review)
