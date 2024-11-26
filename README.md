# 📰 Fake News Detection Using Machine Learning  

Fake news has become a pervasive issue in today's digital age, influencing public opinion and spreading misinformation rapidly. This project aims to tackle this problem by building a machine learning model capable of distinguishing between real and fake news articles.

---

## 📋 Table of Contents
- [About the Project](#about-the-project)  
- [Dataset](#dataset)  
- [Approach](#approach)  
- [Transformer Experimentation](#transformer-experimentation)  
- [Installation and Usage](#installation-and-usage)  
- [Results](#results)   

---

## 📖 About the Project
This project implements a machine learning pipeline for fake news detection using **Logistic Regression** for binary classification. Additionally, we experimented with **transformer-based models** (e.g., DistilBERT) but faced computational challenges during training. The primary goals were:  
- Achieving high recall to maximize the detection of fake news.  
- Reducing false positives for increased reliability.  

---

## 📊 Dataset
The dataset consists of **44,919 news articles**, with the following features:
- **title**: Headline of the news.  
- **text**: Main body of the news article.  
- **subject**: Category of the news (e.g., politics, technology).  
- **class**: Label for fake news detection (`0` = Real, `1` = Fake).  

### Challenges:
- Class imbalance in the dataset.  
- Variability in writing styles across articles.  

Dataset preprocessing included tokenization, removal of stopwords, and feature extraction using **TF-IDF Vectorizer**.

---

## 🛠 Approach

### 1. **Preprocessing**:
- Tokenization and removal of stopwords to clean the textual data.  
- Vectorization using **TF-IDF** to convert text into numerical features.  

### 2. **Model Implementation**:
- **Logistic Regression**:  
  - A lightweight and efficient binary classifier.  
  - Maps text features to probabilities to classify articles as real or fake.  
- Focused on optimizing recall while minimizing false positives.  

---

## 🤖 Transformer Experimentation
We experimented with a **DistilBERT-based model** for enhanced contextual analysis.  
- **Advantages**:  
  - Superior context understanding and ability to capture nuanced differences in text.  
  - Potential for higher accuracy and reduced false positives compared to Logistic Regression.  
- **Challenges**:  
  - **Memory Limitations**:  
    - Training the model with the available hardware resulted in **out-of-memory errors** due to the large dataset and high dimensionality.  
  - **Training Time**:  
    - Computational demands made the process infeasible within the constraints of this project.  

While transformer models remain a promising direction, they were not fully implemented due to hardware constraints.

---

## 🚀 Installation and Usage

### Prerequisites:
- Python 3.7 or above  
- Libraries: `scikit-learn`, `pandas`, `numpy`, `transformers`, `matplotlib`  

### Steps:
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection

---

## 📈 Results

1. Word Cloud of Fake News:
![Screenshot](results/Fake%20Word%20Cloud.png)

2. Word Cloud of Real News:
![Screenshot](results/Real%20Word%20Cloud.png)

3. Classification Report of Logistic Regression Model
![Screenshot](results/Classification%20Report.png)

4. Confusion Matrix of Decision Tree Model
![Screenshot](results/Confusion%20Matrix.png)