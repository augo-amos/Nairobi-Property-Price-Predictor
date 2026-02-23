# Nairobi Property Price Predictor

A machine learning-powered web application that predicts property prices in Nairobi, Kenya based on location, size, bedrooms, and other features.

## Live Demo
[Add your Streamlit Cloud link here after deployment]

## Project Overview

This project scrapes real property listings from multiple Kenyan real estate websites, cleans and processes the data, trains machine learning models, and deploys an interactive web app for price prediction.

### Features
- **Data Collection**: Scrapes 400+ listings from 4 major property sites
- **Data Cleaning**: Handles missing values, standardizes formats
- **Machine Learning**: Multiple models (Linear Regression, Random Forest) with comparison
- **Web App**: Interactive Streamlit interface for price prediction
- **Market Insights**: Visualizations of price trends by location

## Tech Stack
- **Python 3.9+**: Core programming language
- **Pandas/NumPy**: Data processing
- **Scikit-learn/XGBoost**: Machine learning
- **Streamlit**: Web application
- **Plotly**: Interactive visualizations
- **Git/GitHub**: Version control and deployment

## Project Structure
nairobi-proptech/
├── data/
│ ├── raw/ # Raw scraped data
│ └── processed/ # Cleaned and feature-engineered data
├── models/ # Trained model files
├── notebooks/ # Jupyter notebooks for analysis
├── scripts/ # Python scripts for data processing
├── app/ # Streamlit web application
├── dashboard/ # Dashboard files
├── presentation/ # Presentation materials
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## Installation

### Prerequisites
- Python 3.9 or higher
- Git
- (Optional) Conda for environment management

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/nairobi-proptech.git
cd nairobi-proptech