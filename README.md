# 📊 AI Data Beautifier Pro — Enhanced Analytics & Forecasting

A comprehensive Streamlit application for data analysis, visualization, and forecasting with AI assistance. This enhanced version includes advanced data quality analysis, multiple forecasting models, geographic visualizations, and interactive charts.

## 🚀 Features

### 📈 **Enhanced Data Quality Analysis**
- **Comprehensive Quality Metrics**: Quality score, missing values, duplicates, data type distribution
- **Outlier Detection**: Multiple methods (IQR, Z-score, Isolation Forest)
- **Statistical Analysis**: Skewness, kurtosis, distribution analysis
- **Interactive Dashboard**: Visual quality reports with detailed breakdowns

### 🎨 **Advanced Visualization**
- **Basic Charts**: Bar, Line, Scatter, Histogram with click-to-filter
- **Advanced Charts**: 
  - 3D Scatter Plots
  - Correlation Heatmaps
  - Custom Pivot Table Heatmaps
  - Box and Violin Plots for distribution analysis
- **Geographic Visualization**: Interactive maps with coordinate data
- **AI-Assisted Chart Generation**: Natural language chart creation

### 🔮 **Multiple Forecasting Models**
- **Holt-Winters**: Exponential smoothing with trend and seasonality
- **Prophet**: Facebook's forecasting tool (automatic seasonality detection)
- **ARIMA**: Auto-regressive integrated moving average models
- **Ensemble Forecasting**: Combines multiple models for better accuracy
- **Model Comparison**: Side-by-side comparison of different forecasting approaches

### 🤖 **AI-Powered Features**
- **Smart Chart Planning**: AI generates chart recommendations based on data and user requests
- **Automated Insights**: AI-generated analysis and insights for visualizations
- **Natural Language Processing**: Describe charts in plain English

### 🔍 **Data Research & Generation**
- **Stock Market Data**: Real-time stock data using yfinance
- **Weather Data**: Simulated weather data for analysis
- **Population Data**: Historical population data for countries
- **Synthetic Datasets**: Generate realistic datasets for research (Sales, Healthcare, Education, Finance, E-commerce)
- **Web Scraping**: Extract data from web tables
- **Data Integration**: Use scraped/generated data directly in the main application

### 🔧 **Data Processing**
- **Multi-format Support**: CSV and Excel files
- **Data Merging**: Intelligent merging of multiple datasets
- **Data Cleaning**: Column name cleaning, duplicate removal, missing value handling
- **Export Options**: Download beautified data as CSV or Excel

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd data_beautifier
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API** (optional, for AI features):
   Create `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key-here"
   ```

4. **Run the application**:
   ```bash
   streamlit run data_beautifier.py
   ```

## 📋 Requirements

- Python 3.8+
- Streamlit 1.28.0+
- Pandas 2.0.0+
- Plotly 5.15.0+
- Statsmodels 0.14.0+
- Scikit-learn 1.3.0+
- Prophet 1.1.4+ (optional)
- Requests 2.31.0+ (for web scraping)
- BeautifulSoup4 4.12.0+ (for web scraping)
- yfinance 0.2.18+ (for stock data)
- lxml 4.9.0+ (for HTML parsing)

## 🎯 Usage Guide

### 1. **Data Upload**
- Upload one or more CSV/Excel files
- The app automatically detects and merges datasets with common columns
- Preview your data before processing

### 2. **Data Beautification**
- Clean column names automatically
- Remove duplicates
- Handle missing values with various strategies
- Sort data by any column
- View comprehensive KPI summary

### 3. **Data Quality Analysis**
- Get an overall quality score (0-100)
- Analyze missing values patterns
- Detect outliers using multiple methods
- View statistical distributions
- Examine data type distributions

### 4. **Visualization**
- **Manual Charts**: Choose chart type, X/Y axes, and generate interactive plots
- **AI-Assisted**: Describe your desired chart in natural language
- **Advanced Charts**: Create 3D plots, heatmaps, and distribution analysis
- **Geographic**: Visualize location-based data on interactive maps

### 5. **Forecasting**
- Select from multiple forecasting models
- Choose time and target columns
- Set forecast horizon and frequency
- Compare different model predictions
- View ensemble forecasts combining multiple models

### 6. **Data Research & Generation**
- **Stock Data**: Get real-time stock market data
- **Weather Data**: Generate simulated weather datasets
- **Population Data**: Historical population trends
- **Synthetic Data**: Create realistic datasets for testing
- **Web Scraping**: Extract data from web tables
- **Data Integration**: Use scraped data in main analysis

### 7. **Export Results**
- Download beautified data as CSV or Excel
- Export charts and visualizations
- Save forecasting results

## 🔍 Key Features Explained

### **Data Quality Dashboard**
The enhanced quality analysis provides:
- **Quality Score**: Overall data health (0-100)
- **Missing Values Analysis**: Patterns and percentages
- **Outlier Detection**: Multiple statistical methods
- **Distribution Analysis**: Skewness, kurtosis, and statistical measures

### **Advanced Forecasting**
- **Holt-Winters**: Best for data with trend and seasonality
- **Prophet**: Handles multiple seasonality patterns automatically
- **ARIMA**: Good for stationary time series
- **Ensemble**: Combines predictions for improved accuracy

### **Geographic Visualization**
- Automatically detects latitude/longitude columns
- Interactive scatter maps with color and size encoding
- Support for location-based data analysis

### **AI Chart Generation**
- Describe charts in natural language
- AI suggests appropriate chart types and configurations
- Automated insights and analysis

### **Data Research & Generation**
- **Real-time Stock Data**: Live market data using yfinance
- **Weather Simulation**: Realistic weather datasets for analysis
- **Population Trends**: Historical demographic data
- **Synthetic Datasets**: Realistic data for research and testing
- **Web Scraping**: Extract data from HTML tables
- **Seamless Integration**: Use scraped data directly in analysis

## 🎨 Chart Types Available

### **Basic Charts**
- Bar Charts
- Line Charts
- Scatter Plots
- Histograms

### **Advanced Charts**
- 3D Scatter Plots
- Correlation Heatmaps
- Custom Pivot Table Heatmaps
- Box Plots
- Violin Plots
- Geographic Maps

## 📊 Forecasting Models

### **Holt-Winters Exponential Smoothing**
- Trend: Additive or Multiplicative
- Seasonality: Additive or Multiplicative
- Seasonal Periods: Configurable (0, 7, 12, 4)

### **Prophet (Facebook)**
- Automatic seasonality detection
- Holiday effects
- Trend changes
- Uncertainty intervals

### **ARIMA**
- Auto-regressive (AR) component
- Integrated (I) component for stationarity
- Moving Average (MA) component
- Automatic parameter selection

### **Ensemble**
- Combines multiple model predictions
- Reduces individual model bias
- Improves overall forecast accuracy

## 🔧 Configuration

### **OpenAI Integration**
To enable AI features, add your OpenAI API key to `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-your-api-key-here"
```

### **Customization**
- Modify chart themes and colors
- Adjust forecasting parameters
- Customize data quality thresholds
- Add new chart types

## 🚀 Performance Tips

- **Large Datasets**: Use data sampling for initial exploration
- **Memory Management**: Process data in chunks for very large files
- **Caching**: The app uses Streamlit caching for expensive operations
- **Forecasting**: Use appropriate model based on data characteristics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web app framework
- **Plotly**: For interactive visualizations
- **Prophet**: Facebook's forecasting library
- **Statsmodels**: For statistical analysis and forecasting
- **OpenAI**: For AI-powered features

## 📞 Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**Made with ❤️ for data enthusiasts and analysts**
