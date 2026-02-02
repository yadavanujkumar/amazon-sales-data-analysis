"""
Amazon Sales Data Analysis - Exploratory Data Analysis and Machine Learning Models
This script performs comprehensive EDA and builds predictive ML models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

class AmazonSalesAnalysis:
    """
    Class to perform EDA and build ML models on Amazon Sales dataset
    """
    
    def __init__(self, data_path='amazon_sales_dataset.csv'):
        """Initialize with dataset path"""
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load the dataset"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        return self.df
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n" + "=" * 80)
        print("BASIC DATASET INFORMATION")
        print("=" * 80)
        
        print("\n--- First Few Rows ---")
        print(self.df.head())
        
        print("\n--- Dataset Info ---")
        print(self.df.info())
        
        print("\n--- Statistical Summary ---")
        print(self.df.describe())
        
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found!")
        else:
            print(missing[missing > 0])
        
        print("\n--- Data Types ---")
        print(self.df.dtypes)
        
        print("\n--- Unique Values per Column ---")
        for col in self.df.columns:
            print(f"{col}: {self.df[col].nunique()} unique values")
    
    def exploratory_analysis(self):
        """Perform comprehensive exploratory data analysis"""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        # Categorical columns analysis
        print("\n--- Categorical Columns Analysis ---")
        categorical_cols = ['product_category', 'customer_region', 'payment_method']
        
        for col in categorical_cols:
            print(f"\n{col} Distribution:")
            print(self.df[col].value_counts())
        
        # Numerical columns analysis
        print("\n--- Numerical Columns Analysis ---")
        numerical_cols = ['price', 'discount_percent', 'quantity_sold', 'rating', 
                         'review_count', 'discounted_price', 'total_revenue']
        
        for col in numerical_cols:
            print(f"\n{col} Statistics:")
            print(f"  Mean: {self.df[col].mean():.2f}")
            print(f"  Median: {self.df[col].median():.2f}")
            print(f"  Std Dev: {self.df[col].std():.2f}")
            print(f"  Min: {self.df[col].min():.2f}")
            print(f"  Max: {self.df[col].max():.2f}")
        
        # Date analysis
        print("\n--- Date Analysis ---")
        self.df['order_date'] = pd.to_datetime(self.df['order_date'])
        print(f"Date Range: {self.df['order_date'].min()} to {self.df['order_date'].max()}")
        
        # Revenue analysis by category
        print("\n--- Revenue by Product Category ---")
        revenue_by_category = self.df.groupby('product_category')['total_revenue'].agg(['sum', 'mean', 'count'])
        print(revenue_by_category.sort_values('sum', ascending=False))
        
        # Revenue analysis by region
        print("\n--- Revenue by Customer Region ---")
        revenue_by_region = self.df.groupby('customer_region')['total_revenue'].agg(['sum', 'mean', 'count'])
        print(revenue_by_region.sort_values('sum', ascending=False))
        
        # Average rating by category
        print("\n--- Average Rating by Product Category ---")
        rating_by_category = self.df.groupby('product_category')['rating'].mean()
        print(rating_by_category.sort_values(ascending=False))
        
        # Correlation analysis
        print("\n--- Correlation Analysis ---")
        correlation = self.df[numerical_cols].corr()
        print(correlation['total_revenue'].sort_values(ascending=False))
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        
        # 1. Distribution of numerical features
        numerical_cols = ['price', 'discount_percent', 'quantity_sold', 'rating', 'total_revenue']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, col in enumerate(numerical_cols):
            axes[idx].hist(self.df[col], bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: distributions.png")
        plt.close()
        
        # 2. Categorical features analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        self.df['product_category'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Product Category Distribution')
        axes[0].set_xlabel('Category')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        self.df['customer_region'].value_counts().plot(kind='bar', ax=axes[1], color='lightgreen')
        axes[1].set_title('Customer Region Distribution')
        axes[1].set_xlabel('Region')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        
        self.df['payment_method'].value_counts().plot(kind='bar', ax=axes[2], color='salmon')
        axes[2].set_title('Payment Method Distribution')
        axes[2].set_xlabel('Payment Method')
        axes[2].set_ylabel('Count')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('categorical_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: categorical_distributions.png")
        plt.close()
        
        # 3. Revenue analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Revenue by category
        revenue_cat = self.df.groupby('product_category')['total_revenue'].sum().sort_values()
        revenue_cat.plot(kind='barh', ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('Total Revenue by Product Category')
        axes[0, 0].set_xlabel('Total Revenue')
        
        # Revenue by region
        revenue_reg = self.df.groupby('customer_region')['total_revenue'].sum().sort_values()
        revenue_reg.plot(kind='barh', ax=axes[0, 1], color='seagreen')
        axes[0, 1].set_title('Total Revenue by Customer Region')
        axes[0, 1].set_xlabel('Total Revenue')
        
        # Average order value by category
        avg_revenue_by_category = self.df.groupby('product_category')['total_revenue'].mean().sort_values()
        avg_revenue_by_category.plot(kind='barh', ax=axes[1, 0], color='coral')
        axes[1, 0].set_title('Average Order Value by Category')
        axes[1, 0].set_xlabel('Average Revenue')
        
        # Quantity sold by category
        qty_cat = self.df.groupby('product_category')['quantity_sold'].sum().sort_values()
        qty_cat.plot(kind='barh', ax=axes[1, 1], color='mediumpurple')
        axes[1, 1].set_title('Total Quantity Sold by Category')
        axes[1, 1].set_xlabel('Total Quantity')
        
        plt.tight_layout()
        plt.savefig('revenue_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: revenue_analysis.png")
        plt.close()
        
        # 4. Correlation heatmap
        numerical_cols_corr = ['price', 'discount_percent', 'quantity_sold', 'rating', 
                               'review_count', 'discounted_price', 'total_revenue']
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df[numerical_cols_corr].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, fmt='.2f')
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: correlation_heatmap.png")
        plt.close()
        
        # 5. Box plots for outlier detection
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, col in enumerate(numerical_cols):
            self.df.boxplot(column=col, ax=axes[idx])
            axes[idx].set_title(f'Box Plot: {col}')
            axes[idx].set_ylabel(col)
        
        plt.tight_layout()
        plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: boxplots.png")
        plt.close()
        
        # 6. Rating distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['rating'], bins=20, edgecolor='black', alpha=0.7, color='gold')
        plt.title('Distribution of Product Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.axvline(self.df['rating'].mean(), color='red', linestyle='--', 
                    label=f"Mean: {self.df['rating'].mean():.2f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig('rating_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: rating_distribution.png")
        plt.close()
        
        # 7. Time series analysis
        # Create a copy to avoid modifying the original dataframe
        df_temp = self.df.copy()
        df_temp['year_month'] = df_temp['order_date'].dt.to_period('M')
        monthly_revenue = df_temp.groupby('year_month')['total_revenue'].sum()
        
        plt.figure(figsize=(14, 6))
        monthly_revenue.plot(kind='line', marker='o', color='navy')
        plt.title('Monthly Revenue Trend')
        plt.xlabel('Month')
        plt.ylabel('Total Revenue')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('monthly_revenue_trend.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: monthly_revenue_trend.png")
        plt.close()
        
        print("\n✓ All visualizations created successfully!")
    
    def preprocess_data(self):
        """Preprocess data for machine learning"""
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING FOR ML")
        print("=" * 80)
        
        # Create a copy for processing
        self.df_processed = self.df.copy()
        
        # Feature engineering - extract date features
        self.df_processed['year'] = self.df_processed['order_date'].dt.year
        self.df_processed['month'] = self.df_processed['order_date'].dt.month
        self.df_processed['day'] = self.df_processed['order_date'].dt.day
        self.df_processed['day_of_week'] = self.df_processed['order_date'].dt.dayofweek
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = ['product_category', 'customer_region', 'payment_method']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df_processed[f'{col}_encoded'] = le.fit_transform(self.df_processed[col])
            label_encoders[col] = le
        
        print("✓ Categorical variables encoded")
        print("✓ Date features extracted")
        print(f"✓ Processed dataset shape: {self.df_processed.shape}")
        
        return self.df_processed
    
    def build_regression_models(self):
        """Build regression models to predict total_revenue"""
        print("\n" + "=" * 80)
        print("BUILDING REGRESSION MODELS (Predicting Total Revenue)")
        print("=" * 80)
        
        # Select features for regression
        feature_cols = ['price', 'discount_percent', 'quantity_sold', 'rating', 
                       'review_count', 'product_category_encoded', 
                       'customer_region_encoded', 'payment_method_encoded',
                       'year', 'month', 'day_of_week']
        
        X = self.df_processed[feature_cols]
        y = self.df_processed['total_revenue']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, 
                                                   random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, 
                                                           random_state=42)
        }
        
        # Train and evaluate models
        results = []
        
        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            
            # Train model
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R² Score: {r2:.4f}")
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R2_Score': r2
            })
            
            self.models[name] = model
        
        # Results DataFrame
        results_df = pd.DataFrame(results).sort_values('R2_Score', ascending=False)
        print("\n" + "=" * 80)
        print("MODEL COMPARISON - REGRESSION")
        print("=" * 80)
        print(results_df.to_string(index=False))
        
        self.results['regression'] = results_df
        
        # Save feature importance for tree-based models
        best_model_name = results_df.iloc[0]['Model']
        if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            best_model = self.models[best_model_name]
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print(f"\n--- Feature Importance ({best_model_name}) ---")
            print(feature_importance.to_string(index=False))
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['Feature'], feature_importance['Importance'])
            plt.xlabel('Importance')
            plt.title(f'Feature Importance - {best_model_name}')
            plt.tight_layout()
            plt.savefig('feature_importance_regression.png', dpi=300, bbox_inches='tight')
            print("\n✓ Saved: feature_importance_regression.png")
            plt.close()
        
        # Plot model comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.barh(results_df['Model'], results_df['RMSE'], color='steelblue')
        plt.xlabel('RMSE (Lower is Better)')
        plt.title('Model Comparison - RMSE')
        plt.tight_layout()
        
        plt.subplot(1, 2, 2)
        plt.barh(results_df['Model'], results_df['R2_Score'], color='seagreen')
        plt.xlabel('R² Score (Higher is Better)')
        plt.title('Model Comparison - R² Score')
        plt.tight_layout()
        
        plt.savefig('model_comparison_regression.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: model_comparison_regression.png")
        plt.close()
        
        return results_df
    
    def build_classification_models(self):
        """Build classification models to predict rating categories"""
        print("\n" + "=" * 80)
        print("BUILDING CLASSIFICATION MODELS (Predicting Rating Category)")
        print("=" * 80)
        
        # Create rating categories: Low (1-2), Medium (3-4), High (5)
        # Using bins to ensure: 1-2 = Low, 3-4 = Medium, 5 = High
        self.df_processed['rating_category'] = pd.cut(
            self.df_processed['rating'], 
            bins=[0, 2.5, 4.5, 5.1], 
            labels=['Low', 'Medium', 'High']
        )
        
        print(f"\nRating Category Distribution:")
        print(self.df_processed['rating_category'].value_counts())
        
        # Select features (excluding rating itself)
        feature_cols = ['price', 'discount_percent', 'quantity_sold', 
                       'review_count', 'total_revenue', 'product_category_encoded', 
                       'customer_region_encoded', 'payment_method_encoded',
                       'year', 'month', 'day_of_week']
        
        X = self.df_processed[feature_cols]
        y = self.df_processed['rating_category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Classifier
        print("\n--- Training Random Forest Classifier ---")
        rf_classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        )
        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred))
        
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Low', 'Medium', 'High'],
                    yticklabels=['Low', 'Medium', 'High'])
        plt.title('Confusion Matrix - Rating Category Prediction')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix_classification.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: confusion_matrix_classification.png")
        plt.close()
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_classifier.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n--- Feature Importance ---")
        print(feature_importance.to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance - Random Forest Classifier')
        plt.tight_layout()
        plt.savefig('feature_importance_classification.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: feature_importance_classification.png")
        plt.close()
        
        self.models['RF_Classifier'] = rf_classifier
        self.results['classification'] = {'accuracy': accuracy}
        
        return accuracy
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 80)
        
        report = []
        report.append("=" * 80)
        report.append("AMAZON SALES DATA ANALYSIS - COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Dataset summary
        report.append("1. DATASET SUMMARY")
        report.append("-" * 80)
        report.append(f"Total Records: {len(self.df):,}")
        report.append(f"Total Features: {len(self.df.columns)}")
        report.append(f"Date Range: {self.df['order_date'].min()} to {self.df['order_date'].max()}")
        report.append(f"Total Revenue: ${self.df['total_revenue'].sum():,.2f}")
        report.append(f"Average Order Value: ${self.df['total_revenue'].mean():.2f}")
        report.append("")
        
        # Key insights
        report.append("2. KEY INSIGHTS")
        report.append("-" * 80)
        
        top_category = self.df.groupby('product_category')['total_revenue'].sum().idxmax()
        top_category_revenue = self.df.groupby('product_category')['total_revenue'].sum().max()
        report.append(f"Top Revenue Category: {top_category} (${top_category_revenue:,.2f})")
        
        top_region = self.df.groupby('customer_region')['total_revenue'].sum().idxmax()
        top_region_revenue = self.df.groupby('customer_region')['total_revenue'].sum().max()
        report.append(f"Top Revenue Region: {top_region} (${top_region_revenue:,.2f})")
        
        avg_rating = self.df['rating'].mean()
        report.append(f"Average Product Rating: {avg_rating:.2f}/5.0")
        
        avg_discount = self.df['discount_percent'].mean()
        report.append(f"Average Discount: {avg_discount:.1f}%")
        
        report.append("")
        
        # Model performance
        report.append("3. MACHINE LEARNING MODEL PERFORMANCE")
        report.append("-" * 80)
        
        if 'regression' in self.results:
            report.append("\nRegression Models (Predicting Total Revenue):")
            report.append(self.results['regression'].to_string(index=False))
        
        if 'classification' in self.results:
            report.append(f"\nClassification Model (Predicting Rating Category):")
            report.append(f"Accuracy: {self.results['classification']['accuracy']:.4f}")
        
        report.append("")
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        with open('analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print("\n✓ Report saved to: analysis_report.txt")
        
        return report_text
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "=" * 80)
        print("STARTING COMPLETE AMAZON SALES DATA ANALYSIS")
        print("=" * 80)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Basic information
        self.basic_info()
        
        # Step 3: Exploratory analysis
        self.exploratory_analysis()
        
        # Step 4: Create visualizations
        self.create_visualizations()
        
        # Step 5: Preprocess data
        self.preprocess_data()
        
        # Step 6: Build regression models
        self.build_regression_models()
        
        # Step 7: Build classification models
        self.build_classification_models()
        
        # Step 8: Generate report
        self.generate_report()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nGenerated Files:")
        print("  - distributions.png")
        print("  - categorical_distributions.png")
        print("  - revenue_analysis.png")
        print("  - correlation_heatmap.png")
        print("  - boxplots.png")
        print("  - rating_distribution.png")
        print("  - monthly_revenue_trend.png")
        print("  - feature_importance_regression.png")
        print("  - model_comparison_regression.png")
        print("  - confusion_matrix_classification.png")
        print("  - feature_importance_classification.png")
        print("  - analysis_report.txt")
        print("=" * 80)


if __name__ == "__main__":
    # Run the complete analysis
    analyzer = AmazonSalesAnalysis('amazon_sales_dataset.csv')
    analyzer.run_complete_analysis()
