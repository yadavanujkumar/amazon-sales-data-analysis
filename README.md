# Amazon Sales Data Analysis

A comprehensive Exploratory Data Analysis (EDA) and Machine Learning project on Amazon sales dataset with 50,000 transaction records.

## 📊 Dataset Overview

The dataset contains **50,000 sales records** with the following features:
- **Order Details**: order_id, order_date, product_id
- **Product Information**: product_category, price, discount_percent
- **Transaction Details**: quantity_sold, discounted_price, total_revenue
- **Customer Information**: customer_region, payment_method
- **Product Feedback**: rating, review_count

**Product Categories**: Books, Fashion, Sports, Electronics, Beauty, Home & Kitchen  
**Customer Regions**: Asia, North America, Middle East, Europe  
**Payment Methods**: UPI, Wallet, Credit Card, Debit Card, Cash on Delivery

## 🔍 Key Findings

- **Total Revenue**: $32,866,573.74
- **Average Order Value**: $657.33
- **Top Revenue Category**: Beauty ($5,550,624.97)
- **Top Revenue Region**: Middle East ($8,301,844.50)
- **Average Product Rating**: 3.00/5.0
- **Average Discount**: 13.3%

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yadavanujkumar/amazon-sales-data-analysis.git
cd amazon-sales-data-analysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Analysis

Execute the complete EDA and ML pipeline:
```bash
python3 eda_and_ml_models.py
```

This will generate:
- **12 visualization files** (PNG format)
- **1 comprehensive analysis report** (TXT format)
- **Trained ML models** with performance metrics

## 📈 Analysis Components

### 1. Exploratory Data Analysis (EDA)

#### Statistical Analysis
- Data structure and information
- Missing value analysis
- Descriptive statistics
- Distribution analysis

#### Visualizations Generated
1. **distributions.png** - Distribution of numerical features
2. **categorical_distributions.png** - Product categories, regions, payment methods
3. **revenue_analysis.png** - Revenue breakdown by category and region
4. **correlation_heatmap.png** - Correlation matrix of numerical features
5. **boxplots.png** - Outlier detection for all numerical features
6. **rating_distribution.png** - Customer rating distribution
7. **monthly_revenue_trend.png** - Time series analysis of revenue

### 2. Machine Learning Models

#### Regression Models (Predicting Total Revenue)

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| **Random Forest** | 6.12 | 4.13 | **0.9999** |
| **Gradient Boosting** | 7.00 | 5.10 | 0.9998 |
| **Decision Tree** | 14.15 | 10.05 | 0.9993 |
| Lasso Regression | 187.14 | 139.37 | 0.8708 |
| Ridge Regression | 187.25 | 139.43 | 0.8707 |
| Linear Regression | 187.25 | 139.43 | 0.8707 |

**Best Model**: Random Forest Regressor with R² = 0.9999

**Key Features for Prediction**:
- Price (52% importance)
- Quantity Sold (45% importance)
- Discount Percent (3% importance)

#### Classification Model (Predicting Rating Category)

- **Model**: Random Forest Classifier
- **Task**: Predict if product rating is Low (1-2), Medium (2-4), or High (4-5)
- **Accuracy**: 49.87%
- **Top Features**: Price, Total Revenue, Review Count

**Visualizations**:
- feature_importance_regression.png
- model_comparison_regression.png
- confusion_matrix_classification.png
- feature_importance_classification.png

## 📁 Project Structure

```
amazon-sales-data-analysis/
├── amazon_sales_dataset.csv          # Original dataset (50,000 records)
├── eda_and_ml_models.py              # Main analysis script
├── requirements.txt                   # Python dependencies
├── analysis_report.txt                # Comprehensive analysis report
├── README.md                          # Project documentation
│
├── Visualizations (PNG files):
│   ├── distributions.png
│   ├── categorical_distributions.png
│   ├── revenue_analysis.png
│   ├── correlation_heatmap.png
│   ├── boxplots.png
│   ├── rating_distribution.png
│   ├── monthly_revenue_trend.png
│   ├── feature_importance_regression.png
│   ├── model_comparison_regression.png
│   ├── confusion_matrix_classification.png
│   └── feature_importance_classification.png
```

## 🛠️ Technologies Used

- **Python 3.12+**
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Statistical Analysis**: scipy

## 📊 Model Features

### Regression Features
- price
- discount_percent
- quantity_sold
- rating
- review_count
- product_category (encoded)
- customer_region (encoded)
- payment_method (encoded)
- year, month, day_of_week

## 🎯 Business Insights

1. **Revenue Optimization**: Price and quantity sold are the strongest predictors of revenue
2. **Regional Focus**: Middle East generates highest revenue - potential for expansion
3. **Product Strategy**: Beauty products lead in total revenue
4. **Discount Impact**: Discounts show negative correlation with total revenue (-0.14)
5. **Customer Satisfaction**: Average rating of 3.0/5.0 indicates room for improvement

## 📝 Usage Examples

```python
from eda_and_ml_models import AmazonSalesAnalysis

# Initialize analyzer
analyzer = AmazonSalesAnalysis('amazon_sales_dataset.csv')

# Run complete analysis
analyzer.run_complete_analysis()

# Or run specific components
analyzer.load_data()
analyzer.basic_info()
analyzer.exploratory_analysis()
analyzer.create_visualizations()
analyzer.preprocess_data()
analyzer.build_regression_models()
analyzer.build_classification_models()
analyzer.generate_report()
```

## 🔬 Model Performance Details

### Random Forest Regressor (Best Model)
- **RMSE**: 6.12 (extremely low error)
- **R² Score**: 0.9999 (near-perfect fit)
- **MAE**: 4.13 (average error of $4.13 per prediction)

The model accurately predicts total revenue based on transaction features, making it valuable for:
- Revenue forecasting
- Inventory planning
- Pricing strategy optimization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Yadav Anuj Kumar**

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!