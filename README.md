# Machine Learning Challenge Solution
## Overview
This repository contains my solution to the Machine Learning Coding Challenge, aimed at classifying articles into related groups. The challenge is approached as an unsupervised learning problem, focusing on extracting meaningful patterns from a corpus of text articles without relying on the provided unreliable labels.

## Setup and Installation
To run this solution, ensure you have Python 3 installed. The primary libraries used include pandas, numpy, scikit-learn, and nltk among others, as detailed in the "Packages" section of the notebook.
    - To run the notebook, its best to create a python environment
        `python3 -m venv env`
    - Activate the environment in the directory
        `source env/bin/activate`
    - Install all packages from requirements.txt file
        `pip install -r requirements.txt`

    Note: Please add the train.txt.gzip and test.txt.gzip files into the folder as they were excluded due to being too large.

## Data Preparation and Loading
The dataset comprises text articles provided in GZIP format. Custom functions, as shown in the "Load Datasets" section, were used to load and prepare the data for analysis. This preprocessing step is crucial for the subsequent analysis and model training phases.

## Exploratory Data Analysis (EDA)
The "Explore the Data" section of the notebook details the initial analysis performed on the dataset. This includes examining the distribution of characters and words within the articles, which helped in understanding the dataset's characteristics and guided the feature extraction process.

## Methodology
The approach taken involves multiple steps: preprocessing the text data, feature extraction using TF-IDF (Term Frequency-Inverse Document Frequency), and clustering using algorithms like K-Means. This method was chosen based on its effectiveness in identifying latent patterns in text data, making it suitable for unsupervised classification tasks.

## Model Evaluation and Improvements
The evaluation of the model's performance was based on internal clustering metrics, such as silhouette score, given the unsupervised nature of the problem. Suggestions for improving the model's accuracy include experimenting with different numbers of clusters, trying alternative clustering algorithms, and incorporating more sophisticated natural language processing (NLP) techniques for feature extraction.

## Future Work and Adaptability
Adding new classes or topics to the model involves re-training with updated datasets. Continuous learning strategies could be implemented to mitigate the need for frequent re-training, thereby enhancing the model's adaptability to new data.

## Extra questions
1. How do you evaluate the accuracy and correctness of your model(s)?
    - Based on the experiments and the limited time, to evaluate the accuracy and correctness of unsupervised models like KMeans clustering, traditional accuracy metrics used in supervised learning (e.g., precision, recall) are not applicable because we don't have true labels for comparison. Instead, we use different strategies:
    - Silhouette Score is a metric used to measures how similar an object is to its own cluster compared to other clusters. The silhouette scores for our train and test data were close to zero, indicating overlapping clusters. While this doesn't give a direct measure of "accuracy," it offers insight into the separation and cohesion of the clusters.
    - Besides the silhouette score, other internal metrics like the Davies–Bouldin index or Calinski-Harabasz index can also be used to assess the model without external ground truth. These metrics evaluate cluster separation and cohesion.
    - Inspecting a sample of articles from each cluster to assess whether they are thematically consistent can provide a subjective measure of correctness. Additionally, visualizing clusters using techniques like t-SNE can help us visually assess how well-separated and coherent the clusters are.

2. What could you do to improve the accuracy of your model?
    - Given the low silhouette scores indicating that the clusters might not be meaningfully separated, here are strategies to potentially improve our model's performance:
    - Optimize Text Preprocessing steps (e.g., tokenization, stopword removal, stemming, lemmatization) to ensure that the input data to the model captures the essential information without too much noise.
    - The number of clusters (k) significantly influences clustering outcomes. by using methods like the Elbow Method or silhouette analysis to find a more optimal k value.
    - KMeans assumes clusters to be spherical and might not work well for all types of data distributions, by explore algorithms like DBSCAN, which can handle arbitrary-shaped clusters, or hierarchical clustering, which provides a dendrogram to help decide on the number of clusters.
    - Try Advanced Feature Extraction methods for word embeddings like Word2Vec or GloVe, which capture semantic similarities between words. This could lead to more meaningful clusters by focusing on the underlying topics or themes in the text.
    - Apply techniques like PCA (Principal Component Analysis) or t-SNE before clustering to reduce noise and focus on the most informative features. This can sometimes enhance clustering quality by simplifying the data structure.
    - If possible, incorporate domain-specific knowledge into the feature extraction or clustering process. For example, prioritizing certain keywords or using pre-trained models fine-tuned on similar types of text can help tailor the model to our specific dataset.
    - Clustering is often exploratory and iterative. Regularly reviewing and refining based on qualitative analysis and internal evaluation metrics can gradually improve the model's performance.

3. Imagine that there are future requirements that ask you to add a new class/topic/pattern? What would re-training look like? Are there any concerns? How would you avoid these concerns?
    - Firstly, evaluate whether the current model and the number of clusters or topics (for KMeans or LDA, respectively) still fit the expanded dataset. The inclusion of new patterns may necessitate adjusting these parameters.
    Re-training the Model by incorporate the new data that represents the additional class/topic into our dataset and ensure that this data undergoes the same preprocessing and feature extraction steps as the existing data.
    We could use Incremental Learning or Full Re-training for this purpose
    - For some models, like certain clustering algorithms or neural network-based topic models, we might be able to perform incremental learning, adding new data to the model without starting from scratch. However, for many unsupervised models, especially basic implementations of KMeans or LDA, we would likely need to re-train the model on the entire dataset, including both old and new data, to adequately capture the new class/topic.
    - Potential Concerns and How to Address Them fModel Drift: Over time, as new classes or topics are added, the model's understanding of the data may drift from its original configuration. Regularly monitor model performance and re-train as necessary to maintain its relevance.
    - Scalability: Adding new data can significantly increase the dataset's size, potentially impacting the computational cost and time required for re-training. Optimize our data processing and model training pipeline to handle scalability issues.
    - Consistency in Labeling: In unsupervised learning, labels or cluster/topic assignments are not fixed. Adding new data and re-training can change the assignments even for the existing data. To maintain consistency, consider strategies like semi-supervised learning or fixed seed initialization for clustering.
    - Quality Control: Ensure the quality of new data matches the existing dataset. Poor quality or significantly different data can skew the model, leading to inaccurate cluster or topic assignments.
    - Iterative Approach: Adopt an iterative approach to model development, where we would need to regularly evaluate model performance with new data and refine our approach as needed. This helps in adapting to changes in the data or the underlying patterns.
    - Engage Domain Experts: Regular consultation with domain experts can provide valuable insights into the relevance and coherence of the identified clusters or topics, guiding the refinement of our model.

4. How would you serve your model(s) in production?
    - Building and hosting Training piplines using dockerised containers via Airflow/Kubeflow/Metaflow or standalong yaml/python scripts.
    - Log, store, serve model from a model registry ideally via MLflow /  cloud based model stores
    - Use of schedulers build using IaC that can be trigger for training jobs based on model drift or CI/CD.
    - Servie the model via an inference app to accept RestAPI/GraphQL requests.