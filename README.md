# Machine Learning & Data Science Practical Notes

A comprehensive collection of Jupyter notebooks covering fundamental to advanced concepts in machine learning, data science, and statistical analysis. This repository serves as a practical guide with hands-on implementations and real-world examples.

## 📚 Repository Structure

### 🧹 Data Cleaning
Essential preprocessing techniques for data preparation:
- **Data Analysis**: `analysing_data.ipynb`
- **Missing Values Handling**: 
  - `handling_missing_values.ipynb`
  - `handling_missing_values(imputing data).ipynb`
  - `handling_missing_values(scikit value).ipynb`
- **Feature Scaling**:
  - `Feature_Scaling(Normalisation min_max).ipynb`
  - `Feature_Scaling(Standardisation).ipynb`
- **Encoding Techniques**:
  - `Label_Encoding.ipynb`
  - `OneHotEncoding.ipynb`
  - `Ordinal_Encoding.ipynb`
- **Data Quality**:
  - `Handling_Duplicate_Data.ipynb`
  - `Outlier_removal_IQR.ipynb`
  - `Outlier_removal_Z-Score.ipynb`
  - `Outlier.ipynb`
- **Feature Engineering**:
  - `Feature_Selection.ipynb`
  - `Function_Transformer.ipynb`
- **Data Splitting**: `Train_Test_Split.ipynb`

### 📊 Statistics
Statistical foundations for data science:
- **Descriptive Statistics**:
  - `Central_Tendency.ipynb`
  - `Measures_Of_Variablity.ipynb`
  - `percentile&quartiles.ipynb`
  - `skewness.ipynb`
- **Inferential Statistics**:
  - `T-test.ipynb`
  - `z-test.ipynb`
  - `CentralLimitTheorem.ipynb`
- **Correlation Analysis**: `correlation.ipynb`

### 📈 Regression
Supervised learning for continuous target variables:
- **Linear Regression**:
  - `Simple_Linear_Regression.ipynb`
  - `Multiple_Linear_Regression.ipynb`
- **Advanced Regression**:
  - `Polynomial_Regression.ipynb`
  - `Support Vector Machines ( REGRESSION ).ipynb`
  - `K-Nearest Neighbours ( Regression).ipynb`

### 🎯 Classification
Supervised learning for categorical target variables:
- **Core Algorithms**:
  - `Naive_Byes.ipynb`
  - `Support Vector Machines ( Classification ).ipynb`
- **Tree-Based Methods**: `Decision Tree/`
- **Instance-Based Learning**: `KNN/`
- **Linear Models**: `Logistic Regression/`
- **Model Evaluation**:
  - `Cross-Validation.ipynb`
  - `Hyperparameter Tuning.ipynb`

### 🎭 Clustering
Unsupervised learning for pattern discovery:
- **Partitional Clustering**: `K-means Clustering.ipynb`
- **Hierarchical Clustering**: `Agglomerative Hierarchical.ipynb`
- **Density-Based Clustering**: `DBSCAN Clusturing Algorithm.ipynb`
- **Cluster Evaluation**: `SILHOUETTE SCORE.ipynb`

### 🔗 Association Rule Learning
Market basket analysis and pattern mining:
- **Apriori Algorithm**: `Apriori Algorithm ( Association Rule LEARNING ).ipynb`
- **FP-Growth**: `Frequent Pattern Growth Algorithm.ipynb`
- **Ensemble Methods**:
  - `BAGGING ( BAGGING META-ESTIMATOR RANDOM FOREST ).ipynb`
  - `BAGGING ( BAGGING META-ESTIMATOR RANDOM FOREST ) REGRESSION.ipynb`
- **Voting Classifiers**:
  - `Max Voting, Averaging and Weighted Average Voting ( CLASSIFICATION PRACTICAL ).ipynb`
  - `Max Voting, Averaging & Weighted Average Voting ( REGRESSION PRACTICAL).ipynb`

### ⚖️ Regularization
Techniques to prevent overfitting:
- **Ridge & Lasso**: `Lasso_Ridge_Regularisation.ipynb`

## 📂 Datasets

The repository includes several real-world datasets for practice:

| Dataset | Description | Use Cases |
|---------|-------------|-----------|
| `BostonHousing.csv` | Housing prices in Boston | Regression analysis |
| `diabetes.csv` | Diabetes patient data | Classification/Regression |
| `house_price_regression_dataset.csv` | House price prediction | Regression modeling |
| `Salary_Data.csv` | Salary vs experience | Linear regression |
| `LevelvsSal.csv` | Job level vs salary | Correlation analysis |
| `loan.csv` | Loan approval data | Classification |
| `placement.csv` | Student placement data | Classification |
| `groceries.csv` | Market basket data | Association rule mining |
| `IRIS.csv` | Iris flower classification | Multi-class classification |
| `tips.csv` | Restaurant tips data | Statistical analysis |

## 🚀 Getting Started

### Prerequisites
```bash
pip install jupyter pandas numpy scikit-learn matplotlib seaborn
```

### Optional Dependencies
```bash
pip install plotly scipy statsmodels
```

### Running the Notebooks
1. Clone or download this repository
2. Navigate to the project directory
3. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open any notebook file (`.ipynb`) to start learning!

## 📖 Learning Path

### Beginner Track
1. **Start with Statistics**: Understand data fundamentals
2. **Data Cleaning**: Learn preprocessing techniques
3. **Simple Regression**: Begin with linear relationships
4. **Basic Classification**: Explore categorical predictions

### Intermediate Track
1. **Advanced Algorithms**: SVM, Decision Trees, KNN
2. **Model Evaluation**: Cross-validation and hyperparameter tuning
3. **Clustering**: Unsupervised learning techniques
4. **Feature Engineering**: Advanced preprocessing

### Advanced Track
1. **Ensemble Methods**: Bagging, boosting, voting
2. **Regularization**: Prevent overfitting
3. **Association Rules**: Pattern mining
4. **Statistical Testing**: Hypothesis testing

## 🛠️ Tools & Libraries Used

- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Statistical Analysis**: SciPy, Statsmodels
- **Development Environment**: Jupyter Notebook

## 📊 Key Concepts Covered

### Supervised Learning
- Linear and Logistic Regression
- Decision Trees and Random Forests
- Support Vector Machines
- K-Nearest Neighbors
- Naive Bayes

### Unsupervised Learning
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN

### Model Evaluation
- Cross-Validation
- Hyperparameter Tuning
- Performance Metrics

### Data Preprocessing
- Handling Missing Values
- Feature Scaling and Normalization
- Encoding Categorical Variables
- Outlier Detection and Removal

## 🤝 Contributing

This is a learning repository. Feel free to:
- Suggest improvements to existing notebooks
- Add new examples or datasets
- Fix bugs or improve documentation
- Share your own implementations

## 📄 License

This project is open source and available for educational purposes.

## 📞 Contact

For questions or suggestions, please create an issue in this repository.

---

**Happy Learning! 🎓**
