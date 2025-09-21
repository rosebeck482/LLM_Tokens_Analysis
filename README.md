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
- [`romance_sentence_context_vector_clustering.ipynb`](1_sentence_context_vector_clustering&Log_Log_Analysis/romance_sentence_context_vector_clustering.ipynb) - Analysis on Jane Austen's "Sense and Sensibility" (3000 sentences)
- [`Wiki_sentence_context_vector_clustering.ipynb`](1_sentence_context_vector_clustering&Log_Log_Analysis/Wiki_sentence_context_vector_clustering.ipynb) - Analysis on Wikipedia articles (3000 sentences)

**Models Used:**
- sentence-transformers/all-mpnet-base-v2 (768-dim embeddings)
- sentence-transformers/all-distilroberta-v1 (768-dim embeddings)  
- dunzhang/stella_en_1.5B_v5 (8192-dim embeddings, "StoryEmb" model)

**Methods:**
- K-means clustering with elbow method for optimal k selection
- HDBSCAN clustering with UMAP dimensionality reduction
- Log-log analysis of cluster size distributions
- Power-law fitting with linear regression on log-transformed data

**Key Analyses:**
- Sentence count per cluster distribution
- Word count per cluster analysis
- Log-log plots revealing power-law scaling patterns
- Comparison across different embedding models

### 2. TinyLlama Layer Analysis

**Notebooks:**
- [`Romance_Book_TinyLlama_layers_analysis.ipynb`](2_TinyLlama_last_vs_middle_layer_analysis/Romance_Book_TinyLlama_layers_analysis.ipynb) - Analysis on "Sense and Sensibility"
- [`Wiki_TinyLlama_layers_analysis.ipynb`](2_TinyLlama_last_vs_middle_layer_analysis/Wiki_TinyLlama_layers_analysis.ipynb) - Analysis on Wikipedia articles

**Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- 23 total layers analyzed
- Focus on Layer 0 (first), Layer 15 (2/3rd), Layer 22 (last)
- 2048-dimensional embeddings per layer

**Methods:**
- Cosine similarity grouping with threshold-based clustering
- DBSCAN clustering
- K-means clustering with optimal k selection
- HDBSCAN clustering with hyperparameter tuning via Optuna

**Key Analyses:**
- Layer-wise comparison of clustering patterns
- Log-log analysis of cluster distributions per layer
- Investigation of how semantic organization changes across layers

### 3. Sentences Next Token Analysis

**Notebooks:**
- [`Sentences_Father_analysis.ipynb`](3_sentneces_next_token_analysis/Sentences_Father_analysis.ipynb) - Focused analysis on word "father" in "The Brothers Karamazov"
- [`Sentences_histogram&cluster_next_token_prediction.ipynb`](3_sentneces_next_token_analysis/Sentences_histogram&cluster_next_token_prediction.ipynb) - Broader next-token prediction analysis

**Dataset:** 10,000 sentences from "The Brothers Karamazov" by Fyodor Dostoevsky

**Models:**
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 for embeddings
- Next-token prediction probability analysis

**Methods:**
- K-means clustering on sentence embeddings
- Analysis of predicted next-token probabilities
- Grouping sentences by specific word occurrences
- Statistical analysis of token prediction patterns

### 4. Paragraphs Next Token Analysis

**Notebooks:**
- [`Random_Paragraphs_Stella_Kmeans_loglog.ipynb`](4_paragraphs_next_token_analysis/Random_Paragraphs_Stella_Kmeans_loglog.ipynb) - Wikipedia paragraphs with Stella embeddings
- [`Random(Wiki)_Paragraphs_GPT_Next_token_clustering.ipynb`](4_paragraphs_next_token_analysis/Random(Wiki)_Paragraphs_GPT_Next_token_clustering.ipynb) - GPT-2 next-token clustering on Wikipedia
- [`Romance_Paragraphs_GPT_Next_token_clustering.ipynb`](4_paragraphs_next_token_analysis/Romance_Paragraphs_GPT_Next_token_clustering.ipynb) - GPT-2 analysis on romance text
- [`Romance_Paragraphs_Stella_loglog_analysis.ipynb`](4_paragraphs_next_token_analysis/Romance_Paragraphs_Stella_loglog_analysis.ipynb) - Stella embeddings on romance paragraphs

**Models:**
- GPT-2 (gpt2) for next-token prediction and hidden states
- NovaSearch/stella_en_400M_v5 for semantic embeddings

**Methods:**
- Extraction of final-layer hidden states for last tokens
- Clustering by predicted next-token IDs
- K-means clustering with knee-point detection
- Log-log analysis of cluster size distributions

**Datasets:**
- wikipedia-paragraphs dataset from Hugging Face
- Romance novel paragraphs (minimum 40 characters)

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
- Layer-wise analysis to find optimal conceptual representation

**Key Analyses:**
- Comparison of embeddings for words "mammal" vs "bird" with their respective category centroids
- Investigation of which model layers best capture conceptual relationships

### 6. Ontology LLM-Assisted Clustering

**Notebooks:**
- [`Ontology_LLM_analysis_hierarchical.ipynb`](6_ontology_LLM_assisted_clustering/Ontology_LLM_analysis_hierarchical.ipynb) - Hierarchical clustering approach
- [`Ontology_LLM_analysis_kmeans.ipynb`](6_ontology_LLM_assisted_clustering/Ontology_LLM_analysis_kmeans.ipynb) - K-means based approach

**Dataset:** Mammal taxonomy data from mammaldiversity.org
- 6,801 species with taxonomic hierarchy (species, genus, family, order, subclass)

**Models:**
- OpenAI text-embedding-3-large for embeddings
- GPT-4.1 for LLM-assisted cluster refinement

**Methods:**
- Hierarchical clustering with Ward linkage
- K-means clustering (k=167, matching number of families)
- SPEC (Super-point Enhanced Clustering) - edge point refinement
- LACR (LLM-Assisted Cluster Refinement) - GPT-4.1 based reassignment
- Dimensionality reduction with PCA (100 components)

**Key Analyses:**
- Taxonomic purity metrics at different hierarchical levels
- Mean cluster purity by Order, Family, and Genus
- Ontology construction and visualization
- Export to OWL format for knowledge representation

### 7. Density Analysis

**Notebook:** [`Density_analysis.ipynb`](7.Density%20analysis/Density_analysis.ipynb)

**Dataset:** 3,000 sentences from "Sense and Sensibility" (middle portion)

**Models:**
- BERT-base-cased (12 layers, 768-dim)
- Gemma-2B (2,048-dim)
- Gemma-7B (2,560-dim)

**Methods:**
- Two-NN intrinsic dimensionality estimation
- k-NN graph construction with Louvain community detection
- HDBSCAN clustering with parameter tuning
- Relative density calculations (intra-cluster vs overall distances)

**Key Analyses:**
- Intrinsic dimensionality comparison across models
- Community structure and modularity analysis
- Cluster cohesiveness metrics
- Investigation of how model size affects embedding space structure

## Key Findings

The research reveals consistent power-law scaling patterns across different:
- Models (from BERT to GPT to specialized embeddings)
- Datasets (literature vs Wikipedia)
- Granularities (sentences vs paragraphs)
- Tasks (semantic clustering vs next-token prediction)

This suggests a fundamental organizing principle in how LLMs encode and structure semantic information, potentially reflecting underlying properties of natural language itself.