# From LDA to BERTopic: Improving Topic Modeling on the 20 Newsgroups Dataset

## Project Overview

This project extends a previous LDA-based topic modeling project by comparing a traditional topic modeling method, **Latent Dirichlet Allocation (LDA)**, with a modern transformer-based topic modeling method, **BERTopic**.

The project uses the **20 Newsgroups** dataset and evaluates whether the discovered topics align with the original category labels.

## Research Question

Can transformer-based topic modeling improve topic discovery compared with traditional LDA on the 20 Newsgroups dataset?

## Dataset

The dataset is loaded directly from `scikit-learn`:

```python
from sklearn.datasets import fetch_20newsgroups
```

The dataset contains documents from 20 different newsgroup categories. The original labels are not used for training; they are used only for evaluation.

## Methods

### LDA Baseline

- Text cleaning
- Stopword removal
- Lemmatization with spaCy
- Bigram detection with Gensim
- Dictionary and corpus construction
- Coherence-based topic number selection

### BERTopic Extension

- Sentence embeddings with `all-MiniLM-L6-v2`
- UMAP dimensionality reduction
- HDBSCAN clustering
- c-TF-IDF keyword extraction
- English stopword filtering with `CountVectorizer`

## Evaluation Metrics

The models are evaluated using:

- **Coherence score** for LDA topic interpretability
- **NMI** for topic-category information overlap
- **ARI** for clustering similarity
- **Purity** for dominant category concentration
- Topic-category heatmaps for visual analysis

## Main Results

| Model | Topics | NMI | ARI | Purity |
|---|---:|---:|---:|---:|
| LDA | 11 | 0.2839 | 0.1437 | 0.2798 |
| BERTopic | 163 | 0.5459 | 0.3719 | 0.6812 |

The results show that BERTopic achieves stronger alignment with the original 20 Newsgroups categories. LDA is simpler and more interpretable, while BERTopic produces more fine-grained semantic topics.

## Repository Structure

```text
.
├── notebooks/
│   └── lda_bertopic_20newsgroups_project.ipynb
├── figures/
│   ├── lda_coherence.png
│   ├── lda_heatmap.png
│   ├── bertopic_word_scores.png
│   ├── bertopic_distance_map.png
│   ├── bertopic_heatmap_overview.png
│   └── comparison_metrics.png
├── presentation/
│   └── lda_bertopic_project_presentation.pptx
├── requirements.txt
└── README.md
```

## How to Run

The notebook was developed and tested in **Google Colab**.

1. Open `notebooks/lda_bertopic_20newsgroups_project.ipynb` in Google Colab.
2. Run the environment setup cell.
3. Run the notebook from top to bottom.
4. For faster testing, set `USE_SAMPLE = True` in the BERTopic section.
5. For final results, set `USE_SAMPLE = False`.

## Conclusion

This project shows that BERTopic can improve semantic topic discovery compared with LDA on the 20 Newsgroups dataset. BERTopic produces more specific topic clusters and achieves higher NMI, ARI, and purity scores, while LDA remains useful as a simple and interpretable baseline.

