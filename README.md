# LLM Tokens Analysis

This repository contains research conducted in the CRIS Lab (Prof. Venkat Venkatasubramanian) on how large language models (LLMs) encode and organize semantic meaning. The project explores sentence embeddings, clustering methods, and layer-wise representations, showing that semantic clusters follow power-law scaling across different models, datasets, and tasks.

## Research Report

All findings from this research are documented in the report:

**["Embedding Analysis in LLMs: Layer-Wise Semantics, Clustering, and Power-Law Patterns"](Report_Embedding_Analysis_in_LLMs.pdf)**

This report synthesizes results from all experiments in this repository, providing detailed analysis, methodology, and conclusions about how LLMs organize semantic information.

## Directory Structure

```
LLM_Tokens_Analysis/
├── README.md
├── Report_Embedding_Analysis_in_LLMs.pdf
├── 1_sentence_context_vector_clustering&Log_Log_Analysis/
│   ├── romance_sentence_context_vector_clustering.ipynb    [MAIN]
│   ├── Wiki_sentence_context_vector_clustering.ipynb       [MAIN]
│   └── dataset/ plots_romance/ plots_wiki/ previous_experiments/
├── 2_TinyLlama_last_vs_middle_layer_analysis/
│   ├── Romance_Book_TinyLlama_layers_analysis.ipynb        [MAIN]
│   ├── Wiki_TinyLlama_layers_analysis.ipynb               [MAIN]
│   └── dataset/ romance_plots/ wiki_plots/ previous_experiments/
├── 3_sentneces_next_token_analysis/
│   ├── Sentences_Father_analysis.ipynb                     [MAIN]
│   ├── Sentences_histogram&cluster_next_token_prediction.ipynb [MAIN]
│   ├── Sentences_analysis_report.pdf
│   └── dataset/
├── 4_paragraphs_next_token_analysis/
│   ├── Random_Paragraphs_Stella_Kmeans_loglog.ipynb        [MAIN]
│   ├── Random(Wiki)_Paragraphs_GPT_Next_token_clustering.ipynb [MAIN]
│   ├── Romance_Paragraphs_GPT_Next_token_clustering.ipynb  [MAIN]
│   ├── Romance_Paragraphs_Stella_loglog_analysis.ipynb     [MAIN]
│   └── plots_random/ plots_romance/
├── 5_ontology/
│   ├── Ontology.ipynb                                      [MAIN]
│   └── Ontology_report.pdf
├── 6_ontology_LLM_assisted_clustering/
│   ├── Ontology_LLM_analysis_hierarchical.ipynb           [MAIN]
│   ├── Ontology_LLM_analysis_kmeans.ipynb                 [MAIN]
│   └── dataset/ previous/
└── 7.Density analysis/
    └── Density_analysis.ipynb                              [MAIN]
```

## Detailed Experiment Descriptions

### 1. Sentence Context Vector Clustering & Log-Log Analysis

**Notebooks:**
- [`romance_sentence_context_vector_clustering.ipynb`](1_sentence_context_vector_clustering&Log_Log_Analysis/romance_sentence_context_vector_clustering.ipynb) - Analysis on Jane Austen's "Sense and Sensibility" from Project Gutenberg (3000 middle sentences)
- [`Wiki_sentence_context_vector_clustering.ipynb`](1_sentence_context_vector_clustering&Log_Log_Analysis/Wiki_sentence_context_vector_clustering.ipynb) - Analysis on Wikipedia sentences from Kaggle dataset (3000 middle sentences)

**Models Used:**
- uhhlt/story-emb (StoryEmb model, built on Mistral)
- sentence-transformers/all-mpnet-base-v2 (768-dim embeddings)
- sentence-transformers/all-distilroberta-v1 (768-dim embeddings)

**Methods:**
- K-means clustering with elbow method for optimal k selection (k range: 5-705)
- HDBSCAN clustering with UMAP dimensionality reduction
- Hyperparameter tuning using Optuna for HDBSCAN/UMAP parameters
- Log-log analysis of cluster size distributions
- Power-law fitting with linear regression on log10-transformed data

**Key Analyses:**
- Sentence count per cluster distribution
- Word count per cluster analysis
- Log-log plots revealing power-law scaling patterns
- Comparison across different embedding models

### 2. TinyLlama Layer Analysis

**Notebooks:**
- [`Romance_Book_TinyLlama_layers_analysis.ipynb`](2_TinyLlama_last_vs_middle_layer_analysis/Romance_Book_TinyLlama_layers_analysis.ipynb) - Analysis on "Sense and Sensibility" (3000 middle sentences)
- [`Wiki_TinyLlama_layers_analysis.ipynb`](2_TinyLlama_last_vs_middle_layer_analysis/Wiki_TinyLlama_layers_analysis.ipynb) - Analysis on Wikipedia sentences from Kaggle (3000 middle sentences)

**Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Focus on Layer 15 (middle/2/3rd) and Layer 22 (last)
- 2048-dimensional embeddings per layer
- Layer-wise comparison between middle and final representations

**Methods:**
- Cosine similarity grouping with threshold-based clustering
- DBSCAN clustering with cosine distance metric
- K-means clustering with elbow method for optimal k selection (k ranges: 5-705)
- HDBSCAN clustering with UMAP dimensionality reduction
- Hyperparameter tuning using Optuna for HDBSCAN/UMAP parameters
- Cross-layer parameter optimization (using best parameters from one layer on another)

**Key Analyses:**
- Layer-wise comparison of clustering patterns
- Log-log analysis of cluster distributions per layer
- Investigation of how semantic organization changes across layers

### 3. Sentences Next Token Analysis

**Notebooks:**
- [`Sentences_Father_analysis.ipynb`](3_sentneces_next_token_analysis/Sentences_Father_analysis.ipynb) - Focused analysis on word "father" in "The Brothers Karamazov"
- [`Sentences_histogram&cluster_next_token_prediction.ipynb`](3_sentneces_next_token_analysis/Sentences_histogram&cluster_next_token_prediction.ipynb) - Broader next-token prediction analysis

**Dataset:** 10,000 middle sentences from "The Brothers Karamazov" by Fyodor Dostoevsky (Project Gutenberg)

**Models:**
- sentence-transformers/all-mpnet-base-v2 for sentence embeddings
- GPT-2 (gpt2) for next-token prediction and hidden states analysis

**Methods:**
- K-means clustering on sentence embeddings (k=142 for father analysis, k=300-800 range for broader analysis)
- Analysis of predicted next-token probabilities using GPT-2
- Clustering by predicted next-token IDs (novel clustering approach)
- Thematic concentration analysis around specific concepts (e.g., "father")
- Distance analysis (Euclidean distance and cosine similarity to cluster centroids)
- Log-log analysis of cluster size distributions
- Statistical binning and histogram analysis of clustering patterns

### 4. Paragraphs Next Token Analysis

**Notebooks:**
- [`Random_Paragraphs_Stella_Kmeans_loglog.ipynb`](4_paragraphs_next_token_analysis/Random_Paragraphs_Stella_Kmeans_loglog.ipynb) - Wikipedia paragraphs with Stella embeddings
- [`Random(Wiki)_Paragraphs_GPT_Next_token_clustering.ipynb`](4_paragraphs_next_token_analysis/Random(Wiki)_Paragraphs_GPT_Next_token_clustering.ipynb) - GPT-2 next-token clustering on Wikipedia
- [`Romance_Paragraphs_GPT_Next_token_clustering.ipynb`](4_paragraphs_next_token_analysis/Romance_Paragraphs_GPT_Next_token_clustering.ipynb) - GPT-2 analysis on romance text
- [`Romance_Paragraphs_Stella_loglog_analysis.ipynb`](4_paragraphs_next_token_analysis/Romance_Paragraphs_Stella_loglog_analysis.ipynb) - Stella embeddings on romance paragraphs

**Models:**
- GPT-2 (gpt2) for next-token prediction and hidden states
- dunzhang/stella_en_400M_v5 for semantic embeddings

**Methods:**
- Extraction of final-layer hidden states for last tokens in paragraphs
- Clustering by predicted next-token IDs (novel clustering approach, 555 unique tokens sampled)
- K-means clustering with knee-point detection (k ranges: 700-1400, optimal k=1132)
- MiniBatchKMeans for large-scale clustering
- Log-log analysis of cluster size distributions
- Power-law fitting with linear regression on log10-transformed data
- Statistical analysis of predicted token probabilities

**Datasets:**
- wikipedia-paragraphs dataset from Hugging Face (agentlans/wikipedia-paragraphs)
- Romance novel paragraphs from Hugging Face (AlekseyKorshuk/romance-books, minimum 40 characters)

### 5. Ontology Analysis

**Notebook:** [`Ontology.ipynb`](5_ontology/Ontology.ipynb)

**Datasets:**
- Mammal names from willcb/mammal-names (Hugging Face)
- Bird names from thepushkarp/common-bird-names (Kaggle)

**Models:**
- sentence-transformers/all-mpnet-base-v2
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (layer-wise analysis)

**Methods:**
- K-means clustering (k=1 for centroids, k=2 for classification)
- Cosine similarity analysis between concept words and cluster centroids
- PCA, t-SNE, and 3D UMAP visualizations
- Layer-wise analysis across all TinyLlama layers to find optimal conceptual representation
- Last-token embedding extraction from each transformer layer
- Cross-category similarity comparisons (mammal vs bird centroids)

**Key Analyses:**
- Comparison of embeddings for words "mammal" vs "bird" with their respective category centroids
- Investigation of which model layers best capture conceptual relationships
- Semantic clustering validation: concept words align with their respective category clusters
- Layer-wise cosine similarity tracking to identify optimal conceptual layers
- Cross-modal visualization using multiple dimensionality reduction techniques
- Hierarchical semantic structure exploration through embedding space analysis

### 6. Ontology LLM-Assisted Clustering

**Notebooks:**
- [`Ontology_LLM_analysis_hierarchical.ipynb`](6_ontology_LLM_assisted_clustering/Ontology_LLM_analysis_hierarchical.ipynb) - Hierarchical clustering approach
- [`Ontology_LLM_analysis_kmeans.ipynb`](6_ontology_LLM_assisted_clustering/Ontology_LLM_analysis_kmeans.ipynb) - K-means based approach

**Dataset:** Mammal taxonomy data from mammaldiversity.org
- 6,801 species with taxonomic hierarchy (species, genus, family, order, subclass)
- MDD_v2.1_6801species.csv file with comprehensive mammal diversity data

**Models:**
- OpenAI text-embedding-3-large for high-dimensional embeddings
- GPT-4.1 for LLM-assisted cluster refinement and semantic validation

**Methods:**
- Hierarchical clustering with Ward linkage (hierarchical approach)
- K-means clustering (k=167, matching number of families in K-means approach)
- SPEC (Super-point Enhanced Clustering) - 10% edge point refinement (α=0.10)
- LACR (LLM-Assisted Cluster Refinement) - GPT-4.1 based reassignment (β=0.10)
- Dimensionality reduction with PCA (100 components for hierarchical approach)
- Iterative refinement: 5 SPEC iterations, 3 LACR iterations
- Cosine similarity-based cluster candidate selection (top-8 nearest clusters)

**Key Analyses:**
- Taxonomic purity metrics at different hierarchical levels (Order, Family, Genus)
- Mean cluster purity calculations with silhouette score evaluation
- Comprehensive ontology construction with hierarchical visualization
- Export to multiple formats: OWL, JSON, GraphML, CSV for knowledge representation
- Dendrogram visualization and radial tree representations
- UMAP visualization colored by taxonomic orders and families

### 7. Density Analysis

**Notebook:** [`Density_analysis.ipynb`](7.Density%20analysis/Density_analysis.ipynb)

**Dataset:** 3,000 middle sentences from "Sense and Sensibility" (Project Gutenberg)
- Extracted from middle portion of the novel (±1500 sentences from center)
- Preprocessed with Gutenberg boilerplate removal and text normalization

**Models:**
- BERT-base-cased (12 layers, 768-dim, 110M parameters) - CLS token embeddings
- Gemma-2B (google/gemma-2b, 2,048-dim) - mean pooling with attention masking
- Gemma-7B (google/gemma-7b, 2,560-dim) - mean pooling with attention masking

**Methods:**
- Embedding normalization: L2 normalization, centering, and re-normalization
- Two-NN intrinsic dimensionality estimation (Facco et al. 2017)
- k-NN graph construction (k=15) with Louvain community detection
- HDBSCAN clustering with automated parameter tuning (min_cluster_size, min_samples)
- Relative density calculations: intra-cluster vs overall space distances
- Graph-based metrics: modularity, average clustering coefficient, community cohesiveness

**Key Analyses:**
- Intrinsic dimensionality comparison across model architectures and sizes
- Community structure analysis: modularity scores and edge density patterns
- Cluster cohesiveness and relative density metrics
- Model size effects: larger models show decreased modularity, increased connectivity
- Embedding space structure: larger models create more structured spaces with distinct clusters
- Density variance analysis: uniform vs heterogeneous cluster organization

## Key Findings

The research reveals consistent power-law scaling patterns across different:
- Models (from BERT to GPT to specialized embeddings)
- Datasets (literature vs Wikipedia)
- Granularities (sentences vs paragraphs)
- Tasks (semantic clustering vs next-token prediction)

This suggests a fundamental organizing principle in how LLMs encode and structure semantic information, potentially reflecting underlying properties of natural language itself.