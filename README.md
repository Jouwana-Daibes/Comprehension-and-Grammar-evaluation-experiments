#Comprehension and Grammar Evaluation experiments
## Comprehension Assessment Experiments

This repository contains the experimental implementation of the **Comprehension Assessment** module developed for the paper:

> **Computer-Aided Language Learning for Arabic-Speaking Children**

The repository includes the experiments conducted before deployment to evaluate semantic similarity models for automatically assessing Arabic-speaking children's spoken responses.

---

## Repository Structure

```
.
├── Comprehension_evaluation.ipynb
├── LICENSE
├── CITATION.cff
└── README.md
```

---

## Overview

The notebooks evaluate multiple semantic similarity approaches for automatic comprehension assessment by comparing a child's response with one or more reference answers.

The evaluated approaches include:

- TF-IDF
- TF-IDF + Latent Semantic Analysis (LSA)
- AraVec (Arabic Word2Vec)
- Cosine Similarity

Each approach is evaluated using standard classification metrics to determine its suitability for deployment in the proposed Computer-Assisted Language Learning (CALL) framework.

---

## Dataset

The experiments use a text dataset collected from Arabic-speaking children aged **10–14 years**.

The dataset includes:

- Children's transcribed spoken responses
- Story identifiers
- Question identifiers
- Multiple reference answers for each question
- Ground-truth labels used for evaluation

The dataset is maintained in a separate GitHub repository and can be accessed here:

**Dataset Repository:**  
[https://github.com/<your-username>/<dataset-repository>
](https://github.com/NawrasRahhal/WatchAndLearn-Question-Answering-Dataset.git)
Please refer to the dataset repository for the dataset structure, documentation, and usage instructions.
---

## Evaluation Metrics

The experiments report:

- Accuracy
- Precision
- Recall
- F1-score

Threshold optimization is performed to identify the best operating point for each evaluated model.

---

## Running the Notebook

Clone the repository:

```bash
git clone [https://github.com/<your-username>/<repository-name>.git](https://github.com/Jouwana-Daibes/Comprehension-and-Grammar-evaluation-experiments.git)
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open

```
notebook.ipynb
```

and execute all cells sequentially.

---

## Related Publication

This repository accompanies the paper:

> **Computer-Aided Language Learning for Arabic-Speaking Children**

The experiments correspond to the **Comprehension Assessment Evaluation** section of the manuscript.

---

## Citation

If you use this repository in your research, please cite both the associated paper and this repository.

See **CITATION.cff** for citation information.

---

## License

This project is released under the **MIT License**.

See the **LICENSE** file for details.

---

## Authors

- Jouwana Daibes
- Klarien Wassaya
- Masa Jalamneh
- Abualsoud Hanani

Department of Electrical and Computer Engineering  
Faculty of Engineering and Technology  
Birzeit University, Palestine
