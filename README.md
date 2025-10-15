# MIND: Multimodal Integration with Neighbourhood-aware Distributions
This repository hosts the implementation of [MIND](https://www.biorxiv.org/content/10.1101/2025.09.15.676314v1.full.pdf), a VAE based multiomics integration model.

<p align="center"><img src="https://github.com/hwxing3259/multi_o_int/blob/main/examples/multiomics_integration_schematic.png" alt="mult_o_int" width="900px" /></p>

## System requirements
### Sofrware dependency and OS
The package is developed under `Python>=3.12.0`, and requires the following packages
```
matplotlib~=3.70
numpy~=1.26.4
pandas~=2.0.0
torch~=2.2.2
openTSNE~=1.0.2
requests~=2.32.5
zipfile~=3.13
```

The `MIND` package is tested on Mac OS and Ubuntu 16.04 systems.

### Installation Guide
`MIND` can be installed using
```
pip install git+https://github.com/hwxing3259/MIND.git
from MIND import MIND
```

Alternatively, one could directly download the python file [MIND_model.py](https://github.com/hwxing3259/multi_o_int/blob/main/MIND/MIND_model.py) to the current working directory and call
```
from MIND_model import MIND
```

### Demonstration
Here we use the Breast invasive carcinoma (BRCA) dataset from The Cancer Genome Atlas (TCGA) ([Weinstein et al., 2013](https://www.nature.com/articles/ng.2764.pdf)) as an example to demonstrate the proposed method. Preprocessed data can be found [here](https://figshare.com/articles/dataset/Multi_O_Int/30032023).
### Load relavent datasets
```
np.random.seed(31415)
torch.manual_seed(31415)
cancer_type='BRCA'
clinic_data = pd.read_csv('./TCGA_preprocessed/{}/clinic_data.csv'.format(cancer_type), header=0, index_col=0)
RNA_data = pd.read_csv('./TCGA_preprocessed/{}/RNA_data.csv'.format(cancer_type), header=0, index_col=0)
methyl_data = pd.read_csv('./TCGA_preprocessed/{}/meth_data.csv'.format(cancer_type), header=0, index_col=0)
rppa_data = pd.read_csv('./TCGA_preprocessed/{}/rppa_data_imp.csv'.format(cancer_type), header=0, index_col=0)
cna_data = pd.read_csv('./TCGA_preprocessed/{}/cna_data.csv'.format(cancer_type), header=0, index_col=0)
miRNA_data = pd.read_csv('./TCGA_preprocessed/{}/miRNA_data_imp.csv'.format(cancer_type), header=0, index_col=0)
```

### Specify and train the model
```
# specify hyperparameters
emb_dim = 64
lr = 1e-4
epoch = 5000
N = clinic_data.shape[0]

data_dict = {'RNA': RNA_data, 'methyl': methyl_data, 'CNA': cna_data, 'miRNA': miRNA_data, 'RPPA': rppa_data}

test = MIND(data_dict=data_dict, device='cpu', emb_dim=emb_dim)
test.my_train(epoch, lr=lr)
```

### Get embeddings
```
with torch.no_grad():
    z_emb = test.get_embedding()[0].cpu().numpy()
```

### Get reconstrcuted data
```
with torch.no_grad():
    reconstructed_data = test.predict()  
```

### Estimated running time
The code above takes roughly 15 mins to run on a MacBook.

## Instruction for use
User needs ot provide a dictionary of multiomics data. Denote $N$ the total number of patients in the multiomics dataset. Each modality $\mathbf{X}_m$ is a $N\times D_m$ matrix where $D_m$ is the feature diemnsion of the $m$-th modality. For each modality $m$, patients not present in $\mathbf{X}_m$ is represented by a row of $\texttt{NaN}$. 

The model will return a matrix of pateint-specific embeddings of size $N \times D_{emb}$, there $D_{emb}$ is the user-supplied embedding size. For each modality $m$, the trained model also returns $\hat{\mathbf{X}}_m$ containing reconstructions and predictions of both observed and missing data of $\mathbf{X}_m$ . 

## Reproducing numerical examples in the paper
Codes for reproducing the synthetic example: [Link](https://github.com/hwxing3259/MIND/blob/main/examples/synthetic_example.ipynb)

Codes for reproducing the TCGA example: [Link](https://github.com/hwxing3259/MIND/blob/main/examples/TCGA_int_example.ipynb)

Codes for reproducing the CCMA example: [Link](https://github.com/hwxing3259/MIND/blob/main/examples/CCMA_example.ipynb)

Codes for reproducing the CCLE example: [Link](https://github.com/hwxing3259/MIND/blob/main/examples/CCLE_example.ipynb)

Pre-processed datasets and pre-computed outputs can be downloaded from: [Link](https://figshare.com/articles/dataset/Multi_O_Int/30032023)
