import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events

from openai import OpenAI
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet not installed. Run: pip install prophet")



# =========================
# Config
# =========================
st.set_page_config(
    page_title="AI Data Beautifier Pro", 
    layout="wide",
    page_icon="📊"
)

# Clean, minimal CSS styling
st.markdown("""
<style>
    .main-header {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        margin-bottom: 1.5rem;
    }
    
    .stButton > button {
        border-radius: 6px;
        border: 1px solid #dee2e6;
        background-color: #007bff;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #0056b3;
        border-color: #0056b3;
    }
    
    .stExpander {
        border-radius: 6px;
        border: 1px solid #dee2e6;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0px 0px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        color: #495057;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
        border-color: #007bff;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Clean, simple header
st.title("📊 AI Data Beautifier Pro")
st.markdown("**Enhanced Analytics & Forecasting Platform**")

# Simple welcome message
st.info("💡 **New users**: Try the sample dataset to explore all features!")

# =========================
# Session State Initialization
# =========================
if 'manual_chart_type' not in st.session_state:
    st.session_state['manual_chart_type'] = "Bar"
if 'manual_x_axis' not in st.session_state:
    st.session_state['manual_x_axis'] = None
if 'manual_y_axis' not in st.session_state:
    st.session_state['manual_y_axis'] = None
if 'auto_fill_chart' not in st.session_state:
    st.session_state['auto_fill_chart'] = False
if 'recommended_chart' not in st.session_state:
    st.session_state['recommended_chart'] = None
if 'chart_form_key' not in st.session_state:
    st.session_state['chart_form_key'] = 0
if 'ai_recommendations' not in st.session_state:
    st.session_state['ai_recommendations'] = None
if 'clear_recommendation' not in st.session_state:
    st.session_state['clear_recommendation'] = False
if 'forecast_expander_open' not in st.session_state:
    st.session_state['forecast_expander_open'] = False
if 'forecast_recommendations' not in st.session_state:
    st.session_state['forecast_recommendations'] = None
if 'forecast_recommendation' not in st.session_state:
    st.session_state['forecast_recommendation'] = None
if 'auto_fill_forecast' not in st.session_state:
    st.session_state['auto_fill_forecast'] = False
if 'forecast_model' not in st.session_state:
    st.session_state['forecast_model'] = 'Holt-Winters'
if 'forecast_time_col' not in st.session_state:
    st.session_state['forecast_time_col'] = None
if 'forecast_target_col' not in st.session_state:
    st.session_state['forecast_target_col'] = None
if 'forecast_freq' not in st.session_state:
    st.session_state['forecast_freq'] = 'M'
if 'forecast_horizon' not in st.session_state:
    st.session_state['forecast_horizon'] = 6
if 'forecast_season' not in st.session_state:
    st.session_state['forecast_season'] = 12
if 'forecast_form_key' not in st.session_state:
    st.session_state['forecast_form_key'] = 0
if 'enhanced_data' not in st.session_state:
    st.session_state['enhanced_data'] = None

# =========================
# Synthetic Data Generation
# =========================
def create_sample_dataset():
    """Create a comprehensive sample dataset for users to explore"""
    np.random.seed(42)
    
    # Generate dates for the last 2 years
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    # Create sample data
    data = {
        'Date': np.random.choice(dates, 1000),
        'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports'], 1000),
        'Region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 1000),
        'Sales_Amount': np.random.normal(500, 200, 1000),
        'Units_Sold': np.random.poisson(50, 1000),
        'Customer_Satisfaction': np.random.uniform(1, 5, 1000),
        'Marketing_Spend': np.random.exponential(100, 1000),
        'Temperature': np.random.normal(20, 10, 1000),
        'Store_Size': np.random.choice(['Small', 'Medium', 'Large'], 1000),
        'Promotion_Active': np.random.choice([True, False], 1000, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic patterns
    df['Sales_Amount'] = df['Sales_Amount'].abs()  # Ensure positive sales
    df['Customer_Satisfaction'] = df['Customer_Satisfaction'].round(1)
    df['Marketing_Spend'] = df['Marketing_Spend'].round(2)
    df['Temperature'] = df['Temperature'].round(1)
    
    # Add seasonal patterns
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    
    # Add some correlations
    df.loc[df['Promotion_Active'] == True, 'Sales_Amount'] *= 1.3
    df.loc[df['Store_Size'] == 'Large', 'Sales_Amount'] *= 1.2
    df.loc[df['Customer_Satisfaction'] > 4, 'Sales_Amount'] *= 1.1
    
    return df

# =========================
# Error Logging
# =========================
def log_error(error: Exception, context: str = ""):
    """Log errors without exposing technical details to users"""
    error_details = {
        'timestamp': datetime.now().isoformat(),
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        'traceback': traceback.format_exc()
    }
    
    # Log to Streamlit's internal logging (for debugging)
    st.error(f"An error occurred: {error_details['error_type']}")
    
    # You could also log to a file or external service here
    # with open('error_log.txt', 'a') as f:
    #     f.write(json.dumps(error_details) + '\n')
    
    return error_details

# OpenAI client (expects OPENAI_API_KEY in .streamlit/secrets.toml)
@st.cache_resource(show_spinner=False)
def get_openai_client():
    return OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", None))

client = get_openai_client()

# =========================
# Enhanced Data Quality & Cleaning
# =========================
def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """Detect outliers using various methods"""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    elif method == 'zscore':
        # Handle NaN values properly for z-score calculation
        col_data = df[column].dropna()
        if len(col_data) > 0:
            z_scores = np.abs(stats.zscore(col_data))
            outlier_indices = col_data.index[z_scores > 3]
            outliers = df.loc[outlier_indices]
        else:
            outliers = pd.DataFrame()
    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        col_data = df[[column]].dropna()
        if len(col_data) > 0:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(col_data)
            outlier_indices = col_data.index[outlier_labels == -1]
            outliers = df.loc[outlier_indices]
        else:
            outliers = pd.DataFrame()
    else:
        outliers = pd.DataFrame()
    
    return outliers

def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive data quality metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['total_rows'] = len(df)
    metrics['total_columns'] = len(df.columns)
    metrics['missing_values'] = df.isna().sum().sum()
    metrics['missing_percentage'] = (metrics['missing_values'] / (metrics['total_rows'] * metrics['total_columns'])) * 100
    
    # Data type distribution
    dtype_counts = df.dtypes.value_counts()
    # Convert pandas dtypes to strings for JSON serialization
    metrics['data_types'] = {str(k): int(v) for k, v in dtype_counts.to_dict().items()}
    
    # Duplicate analysis
    metrics['duplicate_rows'] = df.duplicated().sum()
    metrics['duplicate_percentage'] = (metrics['duplicate_rows'] / metrics['total_rows']) * 100
    
    # Numeric column analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    metrics['numeric_columns'] = len(numeric_cols)
    
    # Initialize numeric_stats
    metrics['numeric_stats'] = {}
    
    if len(numeric_cols) > 0:
        numeric_stats = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                try:
                    numeric_stats[col] = {
                        'mean': float(col_data.mean()) if not pd.isna(col_data.mean()) else 0.0,
                        'std': float(col_data.std()) if not pd.isna(col_data.std()) else 0.0,
                        'skewness': float(col_data.skew()) if not pd.isna(col_data.skew()) else 0.0,
                        'kurtosis': float(col_data.kurtosis()) if not pd.isna(col_data.kurtosis()) else 0.0,
                        'outliers_iqr': int(len(detect_outliers(df, col, 'iqr'))),
                        'outliers_zscore': int(len(detect_outliers(df, col, 'zscore')))
                    }
                except Exception as e:
                    # Handle any calculation errors gracefully
                    numeric_stats[col] = {
                        'mean': float(col_data.mean()) if not pd.isna(col_data.mean()) else 0.0,
                        'std': float(col_data.std()) if not pd.isna(col_data.std()) else 0.0,
                        'skewness': 0.0,
                        'kurtosis': 0.0,
                        'outliers_iqr': 0,
                        'outliers_zscore': 0
                    }
        metrics['numeric_stats'] = numeric_stats
    
    # Categorical column analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    metrics['categorical_columns'] = len(categorical_cols)
    
    # Initialize categorical_stats
    metrics['categorical_stats'] = {}
    
    if len(categorical_cols) > 0:
        categorical_stats = {}
        for col in categorical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                try:
                    most_common = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None
                    most_common_count = col_data.value_counts().iloc[0] if len(col_data.value_counts()) > 0 else 0
                    
                    categorical_stats[col] = {
                        'unique_values': int(col_data.nunique()),
                        'most_common': str(most_common) if most_common is not None else None,
                        'most_common_count': int(most_common_count)
                    }
                except Exception as e:
                    categorical_stats[col] = {
                        'unique_values': 0,
                        'most_common': None,
                        'most_common_count': 0
                    }
        metrics['categorical_stats'] = categorical_stats
    
    # Overall quality score
    quality_score = 100
    quality_score -= metrics['missing_percentage'] * 0.5
    quality_score -= metrics['duplicate_percentage'] * 0.3
    quality_score = max(0, quality_score)
    metrics['quality_score'] = round(quality_score, 1)
    
    return metrics

def render_data_quality_dashboard(df: pd.DataFrame):
    """Render comprehensive data quality dashboard"""
    st.subheader("🔍 Data Quality Analysis")
    
    metrics = calculate_data_quality_metrics(df)
    
    # Quality score card
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Quality Score", f"{metrics['quality_score']}/100", 
                 delta=f"{metrics['quality_score'] - 50:.1f}" if metrics['quality_score'] > 50 else f"{metrics['quality_score'] - 50:.1f}")
    with col2:
        st.metric("Missing Values", f"{metrics['missing_values']:,}", 
                 delta=f"{metrics['missing_percentage']:.1f}%")
    with col3:
        st.metric("Duplicates", f"{metrics['duplicate_rows']:,}", 
                 delta=f"{metrics['duplicate_percentage']:.1f}%")
    with col4:
        st.metric("Data Types", len(metrics['data_types']))
    
    # Show/hide detailed analysis
    show_detailed_analysis = st.checkbox("🔍 Show Detailed Analysis", value=False, help="View data type distribution, missing values patterns, and detailed statistics")
    
    if show_detailed_analysis:
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔢 Numeric Analysis", "📝 Categorical Analysis", "⚠️ Outliers"])
        
        with tab1:
            # Data type distribution
            # Convert pandas dtypes to strings for JSON serialization
            dtype_values = list(metrics['data_types'].values())
            dtype_names = [str(dtype) for dtype in metrics['data_types'].keys()]
            
            fig = px.pie(values=dtype_values, 
                        names=dtype_names,
                        title="Data Type Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Missing values heatmap
            missing_data = df.isnull()
            if missing_data.any().any():
                # Convert to numeric for proper visualization
                missing_data_numeric = missing_data.astype(int)
                fig = px.imshow(missing_data_numeric.T, 
                               title="Missing Values Heatmap",
                               color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if metrics.get('numeric_stats'):
                numeric_df = pd.DataFrame(metrics['numeric_stats']).T
                st.dataframe(numeric_df, use_container_width=True)
                
                # Skewness and kurtosis visualization
                if len(numeric_df) > 0:
                    try:
                        fig = make_subplots(rows=1, cols=2, subplot_titles=['Skewness', 'Kurtosis'])
                        # Convert to numeric and handle any non-numeric values
                        skewness_values = pd.to_numeric(numeric_df['skewness'], errors='coerce').fillna(0)
                        kurtosis_values = pd.to_numeric(numeric_df['kurtosis'], errors='coerce').fillna(0)
                        
                        fig.add_trace(go.Bar(x=numeric_df.index, y=skewness_values, name='Skewness'), row=1, col=1)
                        fig.add_trace(go.Bar(x=numeric_df.index, y=kurtosis_values, name='Kurtosis'), row=1, col=2)
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create skewness/kurtosis visualization: {e}")
            else:
                st.info("No numeric columns found in the dataset.")
        
        with tab3:
            if metrics.get('categorical_stats'):
                categorical_df = pd.DataFrame(metrics['categorical_stats']).T
                st.dataframe(categorical_df, use_container_width=True)
            else:
                st.info("No categorical columns found in the dataset.")
        
        with tab4:
            if metrics.get('numeric_stats'):
                outlier_col = st.selectbox("Select column for outlier analysis", 
                                         list(metrics['numeric_stats'].keys()))
                method = st.selectbox("Outlier detection method", ['iqr', 'zscore', 'isolation_forest'])
                
                try:
                    outliers = detect_outliers(df, outlier_col, method)
                    st.write(f"Found {len(outliers)} outliers using {method.upper()} method")
                    
                    if len(outliers) > 0:
                        st.dataframe(outliers, use_container_width=True)
                        
                        # Outlier visualization
                        try:
                            fig = px.scatter(df, x=outlier_col, y=outlier_col, 
                                           title=f"Outliers in {outlier_col}",
                                           color=df.index.isin(outliers.index),
                                           color_discrete_map={True: 'red', False: 'blue'})
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create outlier visualization: {e}")
                except Exception as e:
                    st.error(f"Error in outlier detection: {e}")









# =========================
# Enhanced Forecasting Models
# =========================
def check_stationarity(timeseries):
    """Check if time series is stationary"""
    result = adfuller(timeseries.dropna())
    return result[1] <= 0.05  # p-value <= 0.05 means stationary

def prophet_forecast(df: pd.DataFrame, date_col: str, target_col: str, periods: int = 30):
    """Forecast using Facebook Prophet"""
    if not PROPHET_AVAILABLE:
        st.error("Prophet not available. Install with: pip install prophet")
        return None, None
    
    # Prepare data for Prophet
    prophet_df = df[[date_col, target_col]].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df = prophet_df.dropna()
    
    if len(prophet_df) < 10:
        st.warning("Insufficient data for Prophet forecasting (need at least 10 observations)")
        return None, None
    
    # Create and fit model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    
    # Make forecast
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return model, forecast

def arima_forecast(df: pd.DataFrame, date_col: str, target_col: str, periods: int = 30):
    """Forecast using ARIMA"""
    # Prepare time series
    ts_data = df.set_index(date_col)[target_col].dropna()
    
    if len(ts_data) < 10:
        st.warning("Insufficient data for ARIMA forecasting")
        return None, None
    
    # Check stationarity and difference if needed
    is_stationary = check_stationarity(ts_data)
    d = 0 if is_stationary else 1
    
    try:
        # Fit ARIMA model (auto-detect p, q)
        model = ARIMA(ts_data, order=(1, d, 1))
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=periods)
        
        return fitted_model, forecast
    except Exception as e:
        st.error(f"ARIMA forecasting failed: {e}")
        return None, None

def ensemble_forecast(df: pd.DataFrame, date_col: str, target_col: str, periods: int = 30):
    """Combine multiple forecasting models"""
    forecasts = {}
    
    # Holt-Winters
    try:
        ts_data = df.set_index(date_col)[target_col].dropna()
        if len(ts_data) > 10:
            hw_model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=12)
            hw_fit = hw_model.fit()
            forecasts['Holt-Winters'] = hw_fit.forecast(periods)
    except:
        pass
    
    # Prophet
    if PROPHET_AVAILABLE:
        try:
            prophet_model, prophet_forecast = prophet_forecast(df, date_col, target_col, periods)
            if prophet_forecast is not None:
                forecasts['Prophet'] = prophet_forecast['yhat'].tail(periods).values
        except:
            pass
    
    # ARIMA
    try:
        arima_model, arima_forecast = arima_forecast(df, date_col, target_col, periods)
        if arima_forecast is not None:
            forecasts['ARIMA'] = arima_forecast.values
    except:
        pass
    
    if len(forecasts) == 0:
        st.warning("No forecasting models could be fitted")
        return None
    
    # Calculate ensemble (simple average)
    ensemble_values = np.mean(list(forecasts.values()), axis=0)
    forecasts['Ensemble'] = ensemble_values
    
    return forecasts

# =========================
# Advanced Chart Types
# =========================
def create_heatmap(df: pd.DataFrame, correlation: bool = True):
    """Create correlation heatmap or custom heatmap"""
    # Handle duplicate columns
    df = handle_duplicate_columns(df)
    
    if correlation:
        # Correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            st.warning("Need at least 2 numeric columns for correlation heatmap")
            return None
        
        corr_matrix = numeric_df.corr()
        fig = px.imshow(corr_matrix, 
                       text_auto=True,
                       aspect="auto",
                       title="Correlation Matrix",
                       color_continuous_scale='RdBu')
    else:
        # Custom heatmap (pivot table)
        st.info("Select columns for custom heatmap")
        
        # Get suitable columns for pivot table
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Ensure we have suitable columns
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for values")
            return None
        
        if len(categorical_cols) < 2:
            st.warning("Need at least 2 categorical columns for pivot table heatmap")
            return None
        
        # Use categorical columns for index and columns, numeric for values
        x_col = st.selectbox("X-axis (categorical)", categorical_cols)
        y_col = st.selectbox("Y-axis (categorical)", [col for col in categorical_cols if col != x_col])
        value_col = st.selectbox("Values (numeric)", numeric_cols)
        
        try:
            # Validate that the selected columns are suitable for pivoting
            if x_col == y_col:
                st.error("X-axis and Y-axis must be different columns")
                return None
            
            # Check for unique values in index and columns
            unique_x = df[x_col].nunique()
            unique_y = df[y_col].nunique()
            
            if unique_x > 50 or unique_y > 50:
                st.warning(f"Large number of unique values ({unique_x} x {unique_y}). This may create a very large heatmap.")
            
            # Create pivot table with error handling
            pivot_table = df.pivot_table(
                values=value_col, 
                index=y_col, 
                columns=x_col, 
                aggfunc='mean',
                fill_value=0
            )
            
            # Check if pivot table is too large
            if pivot_table.shape[0] > 100 or pivot_table.shape[1] > 100:
                st.warning("Pivot table is very large. Consider selecting columns with fewer unique values.")
            
            fig = px.imshow(pivot_table, 
                           text_auto=True,
                           aspect="auto",
                           title=f"Heatmap: {value_col} by {x_col} vs {y_col}",
                           color_continuous_scale='Viridis')
            
        except Exception as e:
            log_error(e, f"Heatmap pivot table creation failed for columns: {x_col}, {y_col}, {value_col}")
            st.error("Failed to create heatmap. Try selecting different columns or check if the selected columns have suitable data types and values.")
            return None
    
    return fig

def create_3d_scatter(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, color_col: str = None):
    """Create 3D scatter plot"""
    # Handle duplicate columns
    df = handle_duplicate_columns(df)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 3:
        st.warning("Need at least 3 numeric columns for 3D scatter plot")
        return None
    
    fig = px.scatter_3d(df, 
                       x=x_col, 
                       y=y_col, 
                       z=z_col,
                       color=color_col,
                       title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}")
    
    return fig

def create_box_violin_plots(df: pd.DataFrame, x_col: str, y_col: str):
    """Create box and violin plots"""
    # Handle duplicate columns
    df = handle_duplicate_columns(df)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Box Plot', 'Violin Plot'])
    
    # Box plot
    fig.add_trace(go.Box(x=df[x_col], y=df[y_col], name='Box'), row=1, col=1)
    
    # Violin plot
    fig.add_trace(go.Violin(x=df[x_col], y=df[y_col], name='Violin'), row=1, col=2)
    
    fig.update_layout(height=500, title_text=f"Distribution Analysis: {y_col} by {x_col}")
    
    return fig

# =========================
# Original Helper Functions (keeping existing ones)
# =========================

# =========================
# Config
# =========================

# OpenAI client (expects OPENAI_API_KEY in .streamlit/secrets.toml)
@st.cache_resource(show_spinner=False)
def get_openai_client():
    return OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", None))

client = get_openai_client()

# =========================
# Helpers
# =========================
def read_any(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def clean_column_names(cols: List[str]) -> List[str]:
    return [c.strip().replace("\n", " ").replace("\t", " ").replace("_", " ").title() for c in cols]


def kpi_cards(df: pd.DataFrame):
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    total_numeric = len(numeric_cols)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{n_rows:,}")
    col2.metric("Columns", f"{n_cols:,}")
    col3.metric("Numeric Columns", f"{total_numeric}")
    col4.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")


def df_to_download_bytes(df: pd.DataFrame, kind: str = "csv") -> bytes:
    if df is None:
        raise ValueError("DataFrame is None - cannot create download file")
    
    if kind == "csv":
        return df.to_csv(index=False).encode("utf-8")
    elif kind == "xlsx":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Beautified")
        return output.getvalue()
    else:
        raise ValueError("kind must be 'csv' or 'xlsx'")


def summarize_schema(df: pd.DataFrame) -> Dict[str, str]:
    dtypes = df.dtypes
    mapping = {}
    for c in df.columns:
        t = str(dtypes[c])
        if np.issubdtype(dtypes[c], np.number):
            mapping[c] = "number"
        elif np.issubdtype(dtypes[c], np.datetime64):
            mapping[c] = "datetime"
        else:
            mapping[c] = "string"
    return mapping


# =========================
# AI chart planning
# =========================
SYSTEM_INSTRUCTIONS = (
    "You are a data viz planner. Given a schema and a user's request, output a minimal JSON chart plan."
    " Allowed chart_type: 'bar','line','scatter','histogram'. Include keys: chart_type, x, y (optional for histogram),"
    " aggregate (one of 'sum','mean','count','none'), group_by (optional list of columns), trendline (true/false)."
    " Choose only from provided columns. If ambiguous, make a reasonable choice. Reply with ONLY JSON."
)

def analyze_data_for_chart_recommendations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze data and recommend the best charts to create"""
    if client is None:
        return []
    
    schema = summarize_schema(df)
    sample_rows = df.head(10).to_dict(orient="records")
    
    # Analyze data characteristics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Count unique values in categorical columns
    categorical_analysis = {}
    for col in categorical_cols:
        unique_count = df[col].nunique()
        categorical_analysis[col] = {
            'unique_count': unique_count,
            'sample_values': df[col].dropna().unique()[:5].tolist()
        }
    
    prompt = {
        "schema": schema,
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "categorical_analysis": categorical_analysis,
        "total_rows": len(df),
        "sample_data": sample_rows
    }
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a data visualization expert. Analyze the dataset and recommend the 3-5 best charts to create.
                
                Consider:
                1. Data types and relationships
                2. Number of unique values in categorical columns
                3. Time series potential
                4. Distribution analysis potential
                5. Correlation analysis potential
                
                Return a JSON array of chart recommendations with:
                - chart_type: 'bar', 'line', 'scatter', 'histogram', 'heatmap', 'box', 'violin'
                - x: column name for x-axis
                - y: column name for y-axis (if applicable)
                - title: descriptive title
                - description: why this chart would be useful
                - priority: 1-5 (5 being highest priority)
                
                Focus on insights that would be most valuable for understanding the data."""},
                {"role": "user", "content": json.dumps(prompt)}
            ],
            temperature=0.3,
            max_tokens=800,
        )
        
        content = resp.choices[0].message.content.strip()
        
        # Extract JSON array
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end != -1:
            recommendations = json.loads(content[start:end+1])
            # Sort by priority
            recommendations.sort(key=lambda x: x.get('priority', 1), reverse=True)
            return recommendations
        else:
            return []
            
    except Exception as e:
        st.warning(f"Could not generate chart recommendations: {e}")
        return []

def analyze_data_for_forecasting_recommendations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze data and recommend the best forecasting configurations with data validation"""
    if client is None:
        return []
    
    # Comprehensive data analysis with error handling
    try:
        schema = summarize_schema(df)
        sample_rows = df.head(20).to_dict(orient="records")
    except Exception as e:
        log_error(e, "Schema analysis failed")
        return []
    
    # Analyze data characteristics for forecasting
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Check for potential time series data with validation
    time_series_potential = []
    for col in df.columns:
        if any(time_word in col.lower() for time_word in ['date', 'time', 'year', 'month', 'day', 'hour', 'minute']):
            # Validate that the column can be converted to datetime
            try:
                pd.to_datetime(df[col].dropna().head(10), errors='coerce')
                time_series_potential.append(col)
            except Exception:
                continue
    
    # Analyze numeric columns for forecasting suitability
    forecastable_numeric_cols = []
    for col in numeric_cols:
        try:
            # Check if column has enough non-null values and variance
            non_null_count = df[col].dropna().count()
            if non_null_count >= 3:  # Minimum required for forecasting
                variance = df[col].var()
                if variance > 0:  # Some variation in the data
                    forecastable_numeric_cols.append(col)
        except Exception:
            continue
    
    # Analyze data patterns and seasonality potential
    data_analysis = {}
    for col in forecastable_numeric_cols:
        try:
            series = df[col].dropna()
            if len(series) >= 3:
                data_analysis[col] = {
                    'count': int(len(series)),
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'trend': 'increasing' if series.iloc[-1] > series.iloc[0] else 'decreasing' if series.iloc[-1] < series.iloc[0] else 'stable'
                }
        except Exception:
            continue
    
    # Determine appropriate frequencies based on data
    suggested_frequencies = []
    if time_series_potential:
        for time_col in time_series_potential:
            try:
                time_series = pd.to_datetime(df[time_col], errors='coerce').dropna()
                if len(time_series) >= 2:
                    time_diff = time_series.diff().dropna()
                    if len(time_diff) > 0:
                        median_diff = time_diff.median()
                        if median_diff <= pd.Timedelta(days=1):
                            suggested_frequencies.append('D')
                        elif median_diff <= pd.Timedelta(days=7):
                            suggested_frequencies.append('W')
                        elif median_diff <= pd.Timedelta(days=31):
                            suggested_frequencies.append('M')
                        else:
                            suggested_frequencies.append('Q')
            except Exception:
                suggested_frequencies.append('M')  # Default to monthly
    else:
        suggested_frequencies = ['M']  # Default frequency
    
    # Ensure we have at least one frequency
    if not suggested_frequencies:
        suggested_frequencies = ['M']
    
    # Create comprehensive prompt with data validation
    try:
        prompt = {
            "schema": schema,
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "forecastable_numeric_columns": forecastable_numeric_cols,
            "datetime_columns": datetime_cols,
            "time_series_potential": time_series_potential,
            "data_analysis": data_analysis,
            "suggested_frequencies": suggested_frequencies,
            "total_rows": int(len(df)),
            "sample_data": sample_rows,
            "data_quality": {
                "missing_values": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
                "duplicates": int(df.duplicated().sum()),
                "unique_counts": {str(k): int(v) for k, v in {col: df[col].nunique() for col in df.columns}.items()}
            }
        }
    except Exception as e:
        log_error(e, "Prompt creation failed")
        return []
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a forecasting expert. Analyze the dataset and recommend ONLY VALID forecasting configurations that will work with the provided data.

CRITICAL REQUIREMENTS:
1. ONLY recommend time columns that exist in the dataset and can be converted to datetime
2. ONLY recommend target columns that are numeric and have sufficient data (at least 3 non-null values)
3. ONLY recommend frequencies that are appropriate for the data structure
4. ONLY recommend models that are suitable for the data characteristics
5. Validate that recommended parameters will actually work together

DATA VALIDATION RULES:
- time_column: Must be in the provided time_series_potential list
- target_column: Must be in the provided forecastable_numeric_columns list  
- frequency: Must be one of ['D', 'W', 'M', 'Q', 'Y'] and appropriate for the data
- horizon: Should be reasonable (3-24 periods) based on data size
- seasonal_period: Only for Holt-Winters, must be 0, 7, 12, or 4

Return a JSON array of forecasting recommendations with:
- time_column: validated time column from time_series_potential
- target_column: validated target column from forecastable_numeric_columns
- model: recommended forecasting model ('Holt-Winters', 'Prophet', 'ARIMA', 'Ensemble')
- frequency: validated frequency from suggested_frequencies or appropriate default
- horizon: reasonable forecast horizon (3-24)
- seasonal_period: appropriate seasonal period (0, 7, 12, 4)
- title: descriptive title
- description: why this forecasting setup would be useful
- priority: 1-5 (5 being highest priority)
- confidence: 1-5 (5 being highest confidence this will work)

Focus on practical, validated forecasting scenarios that will actually work with the provided data."""},
                {"role": "user", "content": json.dumps(prompt)}
            ],
            temperature=0.2,  # Lower temperature for more consistent, validated recommendations
            max_tokens=1000,
        )
        
        content = resp.choices[0].message.content.strip()
        
        # Extract JSON array with better error handling
        try:
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1:
                json_content = content[start:end+1]
                recommendations = json.loads(json_content)
                
                # Ensure recommendations is a list
                if not isinstance(recommendations, list):
                    recommendations = []
                
                # Additional validation of recommendations
                validated_recommendations = []
                for rec in recommendations:
                    try:
                        # Ensure rec is a dictionary
                        if not isinstance(rec, dict):
                            continue
                            
                        # Validate that recommended columns actually exist and are suitable
                        time_col = rec.get('time_column')
                        target_col = rec.get('target_column')
                        
                        if (time_col in time_series_potential and 
                            target_col in forecastable_numeric_cols and
                            rec.get('frequency') in ['D', 'W', 'M', 'Q', 'Y']):
                            
                            # Add confidence score based on data quality
                            confidence = 5  # Start with high confidence
                            
                            try:
                                # Reduce confidence if data is sparse
                                if df[target_col].dropna().count() < 10:
                                    confidence -= 1
                                
                                # Reduce confidence if time series is short
                                if time_col and df[time_col].dropna().count() < 5:
                                    confidence -= 1
                                
                                # Reduce confidence if high missing values
                                if df[target_col].isnull().sum() / len(df) > 0.3:
                                    confidence -= 1
                            except Exception:
                                confidence = 3  # Default confidence if calculation fails
                            
                            rec['confidence'] = max(1, confidence)
                            validated_recommendations.append(rec)
                    except Exception:
                        continue  # Skip invalid recommendations
                
                # Sort by priority and confidence
                validated_recommendations.sort(key=lambda x: (x.get('priority', 1), x.get('confidence', 1)), reverse=True)
                return validated_recommendations
            else:
                return []
        except json.JSONDecodeError as e:
            log_error(e, "JSON parsing failed in forecasting recommendations")
            return []
        except Exception as e:
            log_error(e, "Recommendation validation failed")
            return []
            
    except Exception as e:
        log_error(e, "AI forecasting recommendations generation failed")
        return []




def plan_chart_with_ai(df: pd.DataFrame, user_request: str) -> Optional[Dict[str, Any]]:
    if client is None:
        st.warning("OpenAI key not set. Add OPENAI_API_KEY to .streamlit/secrets.toml to enable AI charts.")
        return None
    schema = summarize_schema(df)
    sample_rows = df.head(10).to_dict(orient="records")
    prompt = {
        "schema": schema,
        "columns": list(df.columns),
        "user_request": user_request,
    }
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": json.dumps(prompt)}
            ],
            temperature=0.2,
            max_tokens=300,
        )
        content = resp.choices[0].message.content.strip()
        # Extract JSON safely
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            st.error("AI did not return valid JSON. Try rephrasing your request.")
            return None
        plan = json.loads(content[start:end+1])
        # validate fields
        allowed_types = {"bar","line","scatter","histogram"}
        plan["chart_type"] = plan.get("chart_type", "bar")
        if plan["chart_type"] not in allowed_types:
            plan["chart_type"] = "bar"
        for key in ["x","y"]:
            if key in plan and plan[key] not in df.columns:
                plan.pop(key, None)
        agg = plan.get("aggregate", "none")
        if agg not in {"sum","mean","count","none"}:
            plan["aggregate"] = "none"
        group_by = plan.get("group_by")
        if group_by and isinstance(group_by, list):
            # Remove duplicates from group_by and filter valid columns
            valid_group_by = []
            seen_group_by = set()
            for g in group_by:
                if g in df.columns and g not in seen_group_by:
                    valid_group_by.append(g)
                    seen_group_by.add(g)
            plan["group_by"] = valid_group_by
        else:
            plan["group_by"] = []
        plan["trendline"] = bool(plan.get("trendline", False))
        return plan
    except Exception as e:
        st.exception(e)
        return None





def safe_groupby_aggregation(df: pd.DataFrame, group_cols: List[str], y_col: str, operation: str) -> pd.DataFrame:
    """Safely perform groupby aggregation with conflict handling"""
    try:
        # Try standard method first
        if operation == "sum":
            result = df.groupby(group_cols, dropna=False)[y_col].sum()
        elif operation == "mean":
            result = df.groupby(group_cols, dropna=False)[y_col].mean()
        elif operation == "count":
            result = df.groupby(group_cols, dropna=False).size()
        else:
            result = df.groupby(group_cols, dropna=False)[y_col].sum()
        
        # Try to reset index
        try:
            return result.reset_index()
        except ValueError as e:
            if "already exists" in str(e):
                # Handle column conflicts by using to_frame
                if operation == "count":
                    return result.to_frame(name="Count").reset_index()
                else:
                    return result.to_frame(name=y_col).reset_index()
            else:
                raise e
    except Exception as e:
        st.warning(f"Groupby aggregation failed: {e}")
        # Last resort: create a simple grouped dataframe
        try:
            return df.groupby(group_cols, dropna=False)[y_col].sum().to_frame(name=y_col).reset_index()
        except ValueError as e2:
            if "already exists" in str(e2):
                # Ultimate fallback: create dataframe manually
                st.info("Using manual dataframe creation to avoid column conflicts.")
                grouped = df.groupby(group_cols, dropna=False)[y_col].sum()
                # Create dataframe manually to avoid conflicts
                result_data = []
                for idx, value in grouped.items():
                    if isinstance(idx, tuple):
                        row = list(idx) + [value]
                    else:
                        row = [idx, value]
                    result_data.append(row)
                
                # Create column names
                col_names = group_cols + [y_col]
                return pd.DataFrame(result_data, columns=col_names)
            else:
                raise e2

def build_chart_dataframe(df: pd.DataFrame, plan: Dict[str, Any]) -> pd.DataFrame:
    # Apply aggregation if requested
    x = plan.get("x")
    y = plan.get("y")
    group_by = plan.get("group_by", [])
    aggregate = plan.get("aggregate", "none")

    # Cast datetimes if possible for x
    if x and (df[x].dtype == object):
        # try to parse dates
        try:
            df = df.copy()
            df[x] = pd.to_datetime(df[x], errors="ignore")
        except Exception:
            pass

    if aggregate == "none" or (not y and plan["chart_type"] in ["histogram"]):
        return df

    if not x and not group_by:
        # aggregate across entire dataset
        if aggregate == "count":
            return pd.DataFrame({"value": [len(df)]})
        elif y:
            op = {"sum": np.sum, "mean": np.mean}.get(aggregate, np.sum)
            return pd.DataFrame({y: [op(df[y].dropna())]})
        else:
            return df

    group_cols = [c for c in ([x] if x else []) + group_by if c]
    if not group_cols:
        return df

    # Ensure unique column names to avoid conflicts
    df_clean = df.copy()
    existing_cols = set(df_clean.columns)
    
    # Check for potential conflicts with group columns
    conflicting_cols = []
    for col in group_cols:
        if col in existing_cols:
            conflicting_cols.append(col)
    
    # Check for duplicate group columns
    duplicate_group_cols = []
    seen_group_cols = set()
    for col in group_cols:
        if col in seen_group_cols:
            duplicate_group_cols.append(col)
        else:
            seen_group_cols.add(col)
    
    # Remove duplicate group columns
    if duplicate_group_cols:
        st.info(f"⚠️ Duplicate group columns detected: {', '.join(duplicate_group_cols)}. Removing duplicates.")
        group_cols = list(dict.fromkeys(group_cols))  # Remove duplicates while preserving order
    
    if conflicting_cols:
        st.info(f"⚠️ Column conflicts detected: {', '.join(conflicting_cols)}. Using alternative aggregation method.")
    
    if aggregate == "count":
        # Use the safe aggregation function
        out = safe_groupby_aggregation(df_clean, group_cols, y if y else "Count", "count")
        
        if y and "Count" in out.columns:
            # rename for consistency
            out = out.rename(columns={"Count": y})
        return out
    else:
        op = {"sum": np.sum, "mean": np.mean}.get(aggregate, np.sum)
        if not y:
            # choose first numeric col if y missing
            num_cols = df_clean.select_dtypes(include=[np.number]).columns
            y = num_cols[0] if len(num_cols) else None
            if y is None:
                return df_clean
        
        # Use the safe aggregation function
        operation = "sum" if aggregate == "sum" else "mean"
        out = safe_groupby_aggregation(df_clean, group_cols, y, operation)
        
        return out

def handle_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle duplicate column names by adding suffixes"""
    if df.columns.duplicated().any():
        # Get duplicate column names
        duplicated_cols = df.columns[df.columns.duplicated()].unique()
        
        # Create a mapping for renaming
        new_columns = []
        seen_columns = {}
        
        for col in df.columns:
            if col in seen_columns:
                seen_columns[col] += 1
                new_columns.append(f"{col}_{seen_columns[col]}")
            else:
                seen_columns[col] = 0
                new_columns.append(col)
        
        df = df.copy()
        df.columns = new_columns
        
        # Show notification about duplicate columns being handled
        st.info(f"⚠️ Duplicate columns detected and renamed: {', '.join(duplicated_cols)}")
        
    return df


def render_plotly(df_chart: pd.DataFrame, plan: Dict[str, Any]):
    # Handle duplicate columns before creating the chart
    df_chart = handle_duplicate_columns(df_chart)
    
    ct = plan.get("chart_type", "bar")
    x = plan.get("x")
    y = plan.get("y")

    # Convert x and y to strings if they are integers (column indices)
    if isinstance(x, int) and 0 <= x < len(df_chart.columns):
        x = df_chart.columns[x]
    elif isinstance(x, int):
        x = None  # Invalid index
    
    if isinstance(y, int) and 0 <= y < len(df_chart.columns):
        y = df_chart.columns[y]
    elif isinstance(y, int):
        y = None  # Invalid index

    # Update column names if they were changed due to duplicates
    if x and isinstance(x, str) and x not in df_chart.columns:
        # Find the actual column name (might have been renamed)
        x_cols = [col for col in df_chart.columns if isinstance(col, str) and (col.startswith(x) or col == x)]
        x = x_cols[0] if x_cols else None
    
    if y and isinstance(y, str) and y not in df_chart.columns:
        # Find the actual column name (might have been renamed)
        y_cols = [col for col in df_chart.columns if isinstance(col, str) and (col.startswith(y) or col == y)]
        y = y_cols[0] if y_cols else None

    # Create a clean DataFrame for plotting with proper data types
    try:
        # Select only the columns we need for the chart
        columns_to_use = []
        if x:
            columns_to_use.append(x)
        if y:
            columns_to_use.append(y)
        
        # If no specific columns, use all columns
        if not columns_to_use:
            columns_to_use = list(df_chart.columns)
        
        # Create a clean DataFrame with only the needed columns
        plot_df = df_chart[columns_to_use].copy()
        
        # Handle data type issues
        for col in plot_df.columns:
            if plot_df[col].dtype == 'object':
                # Try to convert to numeric if possible, otherwise keep as string
                try:
                    plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                    # If all values became NaN, revert to original
                    if plot_df[col].isna().all():
                        plot_df[col] = df_chart[col]
                except:
                    pass  # Keep as string if conversion fails
        
        # Remove rows with NaN values that might cause issues
        plot_df = plot_df.dropna()
        
        if len(plot_df) == 0:
            st.error("No valid data points found for plotting. Please check your data and column selections.")
            return None
        
        # Create the chart based on type
        if ct == "bar":
            fig = px.bar(plot_df, x=x, y=y)
        elif ct == "line":
            fig = px.line(plot_df, x=x, y=y)
        elif ct == "scatter":
            fig = px.scatter(plot_df, x=x, y=y)
        elif ct == "histogram":
            fig = px.histogram(plot_df, x=x or y)
        else:
            fig = px.bar(plot_df, x=x, y=y)

        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        st.info("Try selecting different columns or check your data types.")
        return None



# Initialize merged dataframe
merged = None


# Simple two-column layout
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv","xlsx"], accept_multiple_files=True)

with col2:
    if st.button("🚀 Load Sample Data", type="primary", use_container_width=True):
        sample_df = create_sample_dataset()
        merged = sample_df
        st.success("✅ Sample dataset loaded!")

all_dfs: List[pd.DataFrame] = []
if uploaded_files:
        for f in uploaded_files:
            try:
                df_i = read_any(f)
                st.caption(f"✅ Loaded {f.name} — shape {df_i.shape}")
                all_dfs.append(df_i)
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")

        # merge if multiple
        if len(all_dfs) > 1:
            st.subheader("🔗 Merge Options")
            common = set(all_dfs[0].columns)
            for d in all_dfs[1:]:
                common &= set(d.columns)
            if common:
                merge_key = st.selectbox("Select column to merge on", sorted(list(common)))
                how = st.selectbox("Merge type", ["outer","left","right","inner"], index=0)
                merged = all_dfs[0]
                for d in all_dfs[1:]:
                    merged = pd.merge(merged, d, on=merge_key, how=how, suffixes=('', '_dup'))
                
                # Handle any remaining duplicate columns after merge
                merged = handle_duplicate_columns(merged)
            else:
                st.warning("No common columns found. Using the first file only.")
                merged = all_dfs[0]
        else:
            merged = all_dfs[0]

        st.subheader("🔍 Original Preview")
        st.dataframe(merged.head(20), use_container_width=True)

# =========================
# Data Processing (only if we have data)
# =========================
if merged is not None:
        # =========================
        # Beautify options
        # =========================
        st.sidebar.header("⚙️ Data Beautification")
        
        ops = st.sidebar.multiselect(
            "Choose operations",
            ["Clean column names","Remove duplicates","Fill missing values","Sort data","KPI summary"],
            default=["Clean column names"]
        )

        dfb = merged.copy()
        if "Clean column names" in ops:
            dfb.columns = clean_column_names(dfb.columns.tolist())
        if "Remove duplicates" in ops:
            dfb = dfb.drop_duplicates()
        if "Fill missing values" in ops:
            strategy = st.sidebar.selectbox("Missing value fill strategy", ["N/A","Zero","Forward fill","Backward fill"], index=0)
            if strategy == "N/A":
                dfb = dfb.fillna("N/A")
            elif strategy == "Zero":
                dfb = dfb.fillna(0)
            elif strategy == "Forward fill":
                dfb = dfb.fillna(method="ffill")
            else:
                dfb = dfb.fillna(method="bfill")
        if "Sort data" in ops and len(dfb.columns) > 0:
            sort_col = st.sidebar.selectbox("Sort by column", dfb.columns)
            asc = st.sidebar.toggle("Ascending", value=True)
            try:
                dfb = dfb.sort_values(by=sort_col, ascending=asc)
            except Exception:
                pass

        st.subheader("✨ Beautified Preview")
        st.dataframe(dfb.head(20), use_container_width=True)

        if "KPI summary" in ops:
            st.subheader("📈 KPI Summary")
            kpi_cards(dfb)
            with st.expander("Detailed describe() stats"):
                st.write(dfb.describe(include="all"))
        
        # =========================
        # Enhanced Data Quality Dashboard
        # =========================
        if st.sidebar.checkbox("🔍 Show Data Quality Analysis", value=True):
            render_data_quality_dashboard(dfb)

        # =========================
        # Visualization (manual + AI)
        # =========================
        st.header("📊 Visualization")
        
        tab_manual, tab_ai, tab_advanced = st.tabs([
            "📈 Manual", 
            "🤖 AI-Assisted", 
            "🎨 Advanced"
        ])

        with tab_manual:
            if merged is not None:
                # AI Chart Recommendations
                if client is not None:
                    with st.expander("🤖 AI Chart Recommendations", expanded=True):
                        # Store recommendations in session state to avoid regenerating
                        if 'ai_recommendations' not in st.session_state:
                            st.session_state['ai_recommendations'] = None
                    
                        if st.button("🎯 Get AI Recommendations", key="get_recommendations_manual"):
                            with st.spinner("Analyzing data for best chart recommendations..."):
                                recommendations = analyze_data_for_chart_recommendations(dfb)
                                st.session_state['ai_recommendations'] = recommendations
                    
                        # Display recommendations if available
                        recommendations = st.session_state.get('ai_recommendations')
                        if recommendations:
                            st.success(f"✨ AI found {len(recommendations)} great chart opportunities!")
                            
                            for i, rec in enumerate(recommendations[:3]):  # Show top 3
                                with st.container():
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.write(f"**{i+1}. {rec.get('title', 'Chart')}**")
                                        st.write(f"*{rec.get('description', '')}*")
                                        st.caption(f"Chart Type: {rec.get('chart_type', 'Unknown')} | Priority: {rec.get('priority', 1)}/5")
                                    with col2:
                                        if st.button(f"📊 Create", key=f"create_rec_manual_{i}"):
                                            # Auto-fill the chart parameters without rerun
                                            st.session_state['recommended_chart'] = rec
                                            st.session_state['auto_fill_chart'] = True
                                            st.session_state['manual_chart_type'] = rec.get('chart_type', 'Bar').title()
                                            st.session_state['manual_x_axis'] = rec.get('x', None)
                                            st.session_state['manual_y_axis'] = rec.get('y', None)
                                            # Force re-render by updating a key
                                            st.session_state['chart_form_key'] = st.session_state.get('chart_form_key', 0) + 1
                            
                            # Show/hide all recommendations toggle - only create if recommendations exist
                            if len(recommendations) > 3:  # Only show toggle if there are more than 3 recommendations
                                # Use a unique key that includes the number of recommendations to ensure uniqueness
                                show_all = st.toggle("📋 Show All Recommendations", key=f"show_all_toggle_manual_{len(recommendations)}")
                                
                                if show_all:
                                    st.subheader("📋 All AI Recommendations")
                                    for i, rec in enumerate(recommendations):
                                        with st.container():
                                            col1, col2 = st.columns([3, 1])
                                            with col1:
                                                st.write(f"**{i+1}. {rec.get('title', 'Chart')}**")
                                                st.write(f"*{rec.get('description', '')}*")
                                                st.caption(f"Chart Type: {rec.get('chart_type', 'Unknown')} | Priority: {rec.get('priority', 1)}/5")
                                            with col2:
                                                if st.button(f"📊 Create", key=f"create_rec_all_manual_{i}"):
                                                    st.session_state['recommended_chart'] = rec
                                                    st.session_state['auto_fill_chart'] = True
                                                    st.session_state['manual_chart_type'] = rec.get('chart_type', 'Bar').title()
                                                    st.session_state['manual_x_axis'] = rec.get('x', None)
                                                    st.session_state['manual_y_axis'] = rec.get('y', None)
                                                    # Force re-render by updating a key
                                                    st.session_state['chart_form_key'] = st.session_state.get('chart_form_key', 0) + 1
                                    st.divider()
                        elif st.session_state.get('ai_recommendations') is not None:
                            # This means recommendations were requested but none found
                            st.info("No specific recommendations found. Try the manual chart creation below.")
            else:
                st.info("Please upload files or generate data to use the visualization features.")
                
                # Manual Chart Creation
                st.subheader("📊 Manual Chart Creation")
                
                # Initialize session state for form values if not exists
                if 'manual_chart_type' not in st.session_state:
                    st.session_state['manual_chart_type'] = "Bar"
                if 'manual_x_axis' not in st.session_state:
                    st.session_state['manual_x_axis'] = None
                if 'manual_y_axis' not in st.session_state:
                    st.session_state['manual_y_axis'] = None
                
                # Check if we have a recommended chart to auto-fill
                if (st.session_state.get('auto_fill_chart', False) and 
                    'recommended_chart' in st.session_state and 
                    not st.session_state.get('clear_recommendation', False)):
                    
                    rec = st.session_state['recommended_chart']
                    default_chart_type = rec.get('chart_type', 'Bar').title()
                    default_x = rec.get('x', None)
                    default_y = rec.get('y', None)
                    
                    # Update session state with recommended values
                    st.session_state['manual_chart_type'] = default_chart_type
                    st.session_state['manual_x_axis'] = default_x
                    st.session_state['manual_y_axis'] = default_y
                    
                    # Show auto-fill notification with option to clear
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.success(f"🎯 Auto-filled with AI recommendation: **{rec.get('title', 'Chart')}**")
                    with col2:
                        if st.button("🔄 Clear", key="clear_recommendation_manual"):
                            # Clear the recommendation without rerun
                            if 'recommended_chart' in st.session_state:
                                del st.session_state['recommended_chart']
                            st.session_state['auto_fill_chart'] = False
                            st.session_state['clear_recommendation'] = True
                            # Reset form values
                            st.session_state['manual_chart_type'] = "Bar"
                            st.session_state['manual_x_axis'] = None
                            st.session_state['manual_y_axis'] = None
                    
                    # Clear the auto-fill flag but keep the recommendation for reference
                    st.session_state['auto_fill_chart'] = False
                else:
                    # Reset clear recommendation flag
                    if st.session_state.get('clear_recommendation', False):
                        st.session_state['clear_recommendation'] = False
            
            # Use session state values or defaults
            default_chart_type = st.session_state['manual_chart_type']
            default_x = st.session_state['manual_x_axis']
            default_y = st.session_state['manual_y_axis']
        
        # Map AI chart types to available chart types
        chart_type_mapping = {
            'bar': 'Bar',
            'line': 'Line', 
            'scatter': 'Scatter',
            'histogram': 'Histogram',
            'heatmap': 'Heatmap',
            'box': 'Box',
            'violin': 'Violin'
        }
        
        # Get available chart types based on what the AI recommended
        available_chart_types = ["Bar", "Line", "Scatter", "Histogram"]
        
        # Add AI-recommended chart types to available options
        if 'ai_recommendations' in st.session_state and st.session_state['ai_recommendations']:
            for rec in st.session_state['ai_recommendations']:
                rec_chart_type = chart_type_mapping.get(rec.get('chart_type', '').lower(), rec.get('chart_type', '').title())
                if rec_chart_type not in available_chart_types:
                    available_chart_types.append(rec_chart_type)
        
        # Ensure the default chart type is in the list
        if default_chart_type not in available_chart_types:
            available_chart_types.append(default_chart_type)
        
        # Get form key for dynamic re-rendering
        form_key = st.session_state.get('chart_form_key', 0)
        
        # Chart type selection with session state
        chart_type_index = available_chart_types.index(default_chart_type) if default_chart_type in available_chart_types else 0
        chart_type = st.selectbox("Chart type", available_chart_types, 
                            index=chart_type_index, key=f"chart_type_select_{form_key}")
        
        # Update session state when chart type changes
        if chart_type != st.session_state['manual_chart_type']:
            st.session_state['manual_chart_type'] = chart_type
        
        # X-axis selection with session state
        x_options = [None] + list(dfb.columns)
        x_index = x_options.index(default_x) if default_x in x_options else 0
        x_col = st.selectbox("X-axis", options=x_options, index=x_index, key=f"x_axis_select_{form_key}")
        
        # Update session state when x-axis changes
        if x_col != st.session_state['manual_x_axis']:
            st.session_state['manual_x_axis'] = x_col
        
        y_col = None
        if chart_type != "Histogram":
            # Y-axis selection with session state
            y_options = [None] + list(dfb.columns)
            y_index = y_options.index(default_y) if default_y in y_options else 0
            y_col = st.selectbox("Y-axis", options=y_options, index=y_index, key=f"y_axis_select_{form_key}")
            
            # Update session state when y-axis changes
            if y_col != st.session_state['manual_y_axis']:
                st.session_state['manual_y_axis'] = y_col
        
        if st.button("Render chart", key="manual_chart_render"):
            plan = {
                "chart_type": chart_type.lower(),
                "x": x_col,
                "y": y_col,
                "aggregate": "none",
                "group_by": []
            }
            df_chart = build_chart_dataframe(dfb, plan)
            fig = render_plotly(df_chart, plan)
            if fig is not None:
                selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=500, override_width="100%")
                st.plotly_chart(fig, use_container_width=True)
            
            # AI-generated summary for manual charts
            if client is not None and st.toggle("🤖 Generate AI Summary", value=True, key="manual_ai_summary"):
                try:
                    # Prepare data for AI analysis
                    chart_data = df_chart.head(50).to_csv(index=False)
                    chart_info = f"Chart Type: {chart_type}, X-axis: {x_col}, Y-axis: {y_col}"
                    
                    prompt = f"""
                    Analyze this chart data and provide a concise, insightful summary.
                    
                    Chart Information: {chart_info}
                    Data: {chart_data}
                    
                    Please provide:
                    1. Key insights about the data patterns
                    2. Notable trends or outliers
                    3. Business implications or observations
                    4. Suggestions for further analysis
                    
                    Keep it concise and actionable.
                    """
                    
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a data analyst. Provide clear, actionable insights from chart data."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=400
                    )
                    
                    st.subheader("🤖 AI Analysis")
                    st.write(resp.choices[0].message.content)
                    
                except Exception as e:
                    st.warning(f"AI summary generation failed: {e}")
            
            if fig is not None and 'selected_points' in locals() and selected_points:
                st.subheader("🔍 Click selection → Filtered rows")
                st.caption(str(selected_points[0]))
                sel = selected_points[0]
                if x_col and x_col in dfb.columns and "x" in sel:
                    filtered = dfb[dfb[x_col] == sel["x"]]
                    st.dataframe(filtered, use_container_width=True)
                elif y_col and y_col in dfb.columns and "y" in sel:
                    filtered = dfb[dfb[y_col] == sel["y"]]
                    st.dataframe(filtered, use_container_width=True)

        with tab_ai:
            if merged is not None:
                user_req = st.text_input("Describe the chart you want (e.g., 'sales by region over time, show sum of sales').")
                
                if st.button("✨ Generate with AI", disabled=(client is None)):
                    plan = plan_chart_with_ai(dfb, user_req)
                    if plan:
                        st.success(f"AI plan: {plan}")
                        df_chart = build_chart_dataframe(dfb, plan)
                        fig = render_plotly(df_chart, plan)
                        if fig is not None:
                            selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=500, override_width="100%")
                            st.plotly_chart(fig, use_container_width=True)

                        # AI insight on aggregated data
                        if client is not None and st.toggle("Generate AI insight for this view", value=True):
                            try:
                                snippet = df_chart.head(50).to_csv(index=False)
                                prompt = (
                                    "Provide concise, bullet-point insights about the following summarized view. "
                                    "Note any trends, outliers, or comparisons.\n\n" + snippet
                                )
                                resp2 = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{"role":"system","content":"You are a helpful data analyst."},
                                              {"role":"user","content":prompt}],
                                    temperature=0.2,
                                    max_tokens=300
                                )
                                st.write(resp2.choices[0].message.content)
                            except Exception as e:
                                st.warning(f"Insight generation failed: {e}")

                        if fig is not None and 'selected_points' in locals() and selected_points:
                            st.subheader("🔍 Click selection → Filtered rows")
                            sel = selected_points[0]
                            x = plan.get("x")
                            y = plan.get("y")
                            if x and "x" in sel and x in dfb.columns:
                                st.write(f"Filter: {x} == {sel['x']}")
                                st.dataframe(dfb[dfb[x] == sel["x"]], use_container_width=True)
                            elif y and "y" in sel and y in dfb.columns:
                                st.write(f"Filter: {y} == {sel['y']}")
                                st.dataframe(dfb[dfb[y] == sel["y"]], use_container_width=True)
            else:
                st.info("Please upload files or generate data to use the AI-assisted visualization features.")
        
        with tab_advanced:
            if merged is not None:
                st.subheader("🎨 Advanced Chart Types")
                
                # Heatmap
                st.write("### 🔥 Correlation Heatmap")
                if st.button("Generate Correlation Heatmap"):
                    fig_heatmap = create_heatmap(dfb, correlation=True)
                    if fig_heatmap:
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # 3D Scatter
                st.write("### 🌐 3D Scatter Plot")
                numeric_cols = dfb.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 3:
                    x_3d = st.selectbox("X-axis (3D)", numeric_cols, key="3d_x")
                    y_3d = st.selectbox("Y-axis (3D)", [col for col in numeric_cols if col != x_3d], key="3d_y")
                    z_3d = st.selectbox("Z-axis (3D)", [col for col in numeric_cols if col not in [x_3d, y_3d]], key="3d_z")
                    color_3d = st.selectbox("Color (optional)", [None] + list(dfb.columns), key="3d_color")
                    
                    if st.button("Generate 3D Scatter"):
                        fig_3d = create_3d_scatter(dfb, x_3d, y_3d, z_3d, color_3d)
                        if fig_3d:
                            st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.warning("Need at least 3 numeric columns for 3D scatter plot")
                
                # Box and Violin plots
                st.write("### 📦 Distribution Analysis")
                if len(dfb.columns) >= 2:
                    x_dist = st.selectbox("X-axis (Distribution)", dfb.columns, key="dist_x")
                    y_dist = st.selectbox("Y-axis (Distribution)", dfb.select_dtypes(include=[np.number]).columns, key="dist_y")
                    
                    if st.button("Generate Distribution Plots"):
                        fig_dist = create_box_violin_plots(dfb, x_dist, y_dist)
                        if fig_dist:
                            st.plotly_chart(fig_dist, use_container_width=True)
                
                # Custom Heatmap
                st.write("### 🎯 Custom Heatmap (Pivot Table)")
                if st.button("Generate Custom Heatmap"):
                    fig_custom_heatmap = create_heatmap(dfb, correlation=False)
                    if fig_custom_heatmap:
                        st.plotly_chart(fig_custom_heatmap, use_container_width=True)
            else:
                st.info("Please upload files or generate data to use the advanced visualization features.")



        # =========================
        # Enhanced Forecasting
        # =========================
        st.header("🔮 Forecasting")
        
        # AI Forecasting Recommendations
        if client is not None:
            with st.expander("🤖 AI Forecasting Recommendations", expanded=True):
                # Store forecasting recommendations in session state
                if 'forecast_recommendations' not in st.session_state:
                    st.session_state['forecast_recommendations'] = None
            
                if st.button("🎯 Get Forecasting Recommendations", key="get_forecast_recommendations"):
                    with st.spinner("Analyzing data for best forecasting configurations..."):
                        recommendations = analyze_data_for_forecasting_recommendations(dfb)
                        st.session_state['forecast_recommendations'] = recommendations
            
            # Display recommendations if available
            recommendations = st.session_state.get('forecast_recommendations')
            if recommendations:
                # Show data quality summary
                st.success(f"✨ AI found {len(recommendations)} validated forecasting opportunities!")
                
                # Data quality summary
                with st.expander("📊 Data Quality Analysis", expanded=False):
                    numeric_cols = dfb.select_dtypes(include=[np.number]).columns.tolist()
                    time_cols = [col for col in dfb.columns if any(time_word in col.lower() for time_word in ['date', 'time', 'year', 'month', 'day'])]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Numeric Columns", len(numeric_cols))
                    with col2:
                        st.metric("Time Columns", len(time_cols))
                    with col3:
                        st.metric("Total Rows", len(dfb))
                    
                    if numeric_cols:
                        st.write("**Forecastable Columns:**")
                        for col in numeric_cols:
                            non_null = dfb[col].dropna().count()
                            missing_pct = (dfb[col].isnull().sum() / len(dfb)) * 100
                            st.write(f"- {col}: {non_null} values ({missing_pct:.1f}% missing)")
                
                for i, rec in enumerate(recommendations[:3]):  # Show top 3
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            # Show confidence indicator
                            confidence = rec.get('confidence', 1)
                            confidence_emoji = "🟢" if confidence >= 4 else "🟡" if confidence >= 3 else "🔴"
                            st.write(f"**{i+1}. {rec.get('title', 'Forecast')}** {confidence_emoji}")
                            st.write(f"*{rec.get('description', '')}*")
                            
                            # Enhanced caption with confidence
                            model_info = f"Model: {rec.get('model', 'Unknown')}"
                            time_info = f"Time: {rec.get('time_column', 'N/A')}"
                            target_info = f"Target: {rec.get('target_column', 'N/A')}"
                            priority_info = f"Priority: {rec.get('priority', 1)}/5"
                            confidence_info = f"Confidence: {confidence}/5"
                            
                            st.caption(f"{model_info} | {time_info} | {target_info} | {priority_info} | {confidence_info}")
                            
                            # Show confidence explanation
                            if confidence < 4:
                                if confidence <= 2:
                                    st.warning("⚠️ Low confidence: This forecast may have limited accuracy due to data quality issues.")
                                else:
                                    st.info("ℹ️ Medium confidence: This forecast should work but may have some limitations.")
                        with col2:
                            if st.button(f"📊 Apply", key=f"apply_forecast_{i}"):
                                # Auto-fill the forecasting parameters
                                st.session_state['forecast_recommendation'] = rec
                                st.session_state['auto_fill_forecast'] = True
                                st.session_state['forecast_model'] = rec.get('model', 'Holt-Winters')
                                st.session_state['forecast_time_col'] = rec.get('time_column', None)
                                st.session_state['forecast_target_col'] = rec.get('target_column', None)
                                st.session_state['forecast_freq'] = rec.get('frequency', 'M')
                                st.session_state['forecast_horizon'] = rec.get('horizon', 6)
                                st.session_state['forecast_season'] = rec.get('seasonal_period', 12)
                                # Force re-render by updating a key
                                st.session_state['forecast_form_key'] = st.session_state.get('forecast_form_key', 0) + 1
                
                # Show/hide all recommendations toggle
                if len(recommendations) > 3:
                    show_all_forecast = st.toggle("📋 Show All Forecasting Recommendations", key=f"show_all_forecast_{len(recommendations)}")
                    
                    if show_all_forecast:
                        st.subheader("📋 All Forecasting Recommendations")
                        for i, rec in enumerate(recommendations):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    # Show confidence indicator
                                    confidence = rec.get('confidence', 1)
                                    confidence_emoji = "🟢" if confidence >= 4 else "🟡" if confidence >= 3 else "🔴"
                                    st.write(f"**{i+1}. {rec.get('title', 'Forecast')}** {confidence_emoji}")
                                    st.write(f"*{rec.get('description', '')}*")
                                    
                                    # Enhanced caption with confidence
                                    model_info = f"Model: {rec.get('model', 'Unknown')}"
                                    time_info = f"Time: {rec.get('time_column', 'N/A')}"
                                    target_info = f"Target: {rec.get('target_column', 'N/A')}"
                                    priority_info = f"Priority: {rec.get('priority', 1)}/5"
                                    confidence_info = f"Confidence: {confidence}/5"
                                    
                                    st.caption(f"{model_info} | {time_info} | {target_info} | {priority_info} | {confidence_info}")
                                    
                                    # Show confidence explanation for low confidence recommendations
                                    if confidence < 3:
                                        st.warning("⚠️ Low confidence: This forecast may have limited accuracy due to data quality issues.")
                                with col2:
                                    if st.button(f"📊 Apply", key=f"apply_forecast_all_{i}"):
                                        st.session_state['forecast_recommendation'] = rec
                                        st.session_state['auto_fill_forecast'] = True
                                        st.session_state['forecast_model'] = rec.get('model', 'Holt-Winters')
                                        st.session_state['forecast_time_col'] = rec.get('time_column', None)
                                        st.session_state['forecast_target_col'] = rec.get('target_column', None)
                                        st.session_state['forecast_freq'] = rec.get('frequency', 'M')
                                        st.session_state['forecast_horizon'] = rec.get('horizon', 6)
                                        st.session_state['forecast_season'] = rec.get('seasonal_period', 12)
                                        # Force re-render by updating a key
                                        st.session_state['forecast_form_key'] = st.session_state.get('forecast_form_key', 0) + 1
                            st.divider()
            elif st.session_state.get('forecast_recommendations') is not None:
                st.info("No specific forecasting recommendations found. Try manual forecasting below.")
        
        # Model selection
        forecast_models = ["Holt-Winters", "Prophet", "ARIMA", "Ensemble"]
        if not PROPHET_AVAILABLE:
            forecast_models = [m for m in forecast_models if m != "Prophet"]
        
        # Check if we have a forecasting recommendation to auto-fill
        if st.session_state.get('auto_fill_forecast', False) and 'forecast_recommendation' in st.session_state:
            rec = st.session_state['forecast_recommendation']
            default_model = st.session_state.get('forecast_model', rec.get('model', 'Holt-Winters'))
            default_time_col = st.session_state.get('forecast_time_col', rec.get('time_column', None))
            default_target_col = st.session_state.get('forecast_target_col', rec.get('target_column', None))
            default_freq = st.session_state.get('forecast_freq', rec.get('frequency', 'M'))
            default_horizon = st.session_state.get('forecast_horizon', rec.get('horizon', 6))
            default_season = st.session_state.get('forecast_season', rec.get('seasonal_period', 12))
            
            # Show auto-fill notification
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(f"🎯 Auto-filled with AI recommendation: **{rec.get('title', 'Forecast')}**")
            with col2:
                if st.button("🔄 Clear", key="clear_forecast_recommendation"):
                    if 'forecast_recommendation' in st.session_state:
                        del st.session_state['forecast_recommendation']
                    st.session_state['auto_fill_forecast'] = False
                    # Clear all forecast session state
                    for key in ['forecast_model', 'forecast_time_col', 'forecast_target_col', 'forecast_freq', 'forecast_horizon', 'forecast_season']:
                        if key in st.session_state:
                            del st.session_state[key]
            
            # Clear the auto-fill flag
            st.session_state['auto_fill_forecast'] = False
        else:
            default_model = st.session_state.get('forecast_model', "Holt-Winters")
            default_time_col = st.session_state.get('forecast_time_col', None)
            default_target_col = st.session_state.get('forecast_target_col', None)
            default_freq = st.session_state.get('forecast_freq', "M")
            default_horizon = st.session_state.get('forecast_horizon', 6)
            default_season = st.session_state.get('forecast_season', 12)
        
        # Get form key for dynamic re-rendering
        forecast_form_key = st.session_state.get('forecast_form_key', 0)
        
        # Safely get the index for model selection
        try:
            model_index = forecast_models.index(default_model) if default_model in forecast_models else 0
        except (ValueError, AttributeError):
            model_index = 0
            default_model = "Holt-Winters"
        
        selected_model = st.selectbox("Select forecasting model", forecast_models, 
                                    index=model_index,
                                    key=f"forecast_model_{forecast_form_key}")
        
        # Initialize expander state
        if 'forecast_expander_open' not in st.session_state:
            st.session_state['forecast_expander_open'] = True
        
        with st.expander("Build a time-series forecast", expanded=st.session_state['forecast_expander_open']):
            # choose time and target with safe index calculation
            time_options = [None] + list(dfb.columns)
            try:
                time_index = time_options.index(default_time_col) if default_time_col in time_options else 0
            except (ValueError, AttributeError):
                time_index = 0
                default_time_col = None
        
        time_col = st.selectbox("Date/Time column", time_options, 
                              index=time_index,
                              key=f"ts_time_{forecast_form_key}")
        
        target_options = [None] + list(dfb.select_dtypes(include=[np.number]).columns)
        try:
            target_index = target_options.index(default_target_col) if default_target_col in target_options else 0
        except (ValueError, AttributeError):
            target_index = 0
            default_target_col = None
        
        target_col = st.selectbox("Target numeric column", target_options, 
                                index=target_index,
                                key=f"ts_target_{forecast_form_key}")
        # Define frequency options and handle invalid default values
        freq_options = ["D", "W", "M", "Q", "Y"]
        freq_labels = ["Day", "Week", "Month", "Quarter", "Year"]
        
        # Safely get the index for default_freq
        try:
            freq_index = freq_options.index(default_freq) if default_freq in freq_options else 0
        except (ValueError, AttributeError):
            freq_index = 0
            default_freq = "M"  # Default to monthly if invalid
        
        freq = st.selectbox("Aggregation frequency", freq_options, 
                          index=freq_index,
                          format_func=lambda x: freq_labels[freq_options.index(x)],
                          help="D=day, W=week, M=month, Q=quarter, Y=year",
                          key=f"ts_freq_{forecast_form_key}")
        horizon = st.slider("Forecast periods", min_value=3, max_value=24, value=default_horizon,
                          key=f"ts_horizon_{forecast_form_key}")
        
        if selected_model == "Holt-Winters":
            # Define seasonal period options and handle invalid default values
            season_options = [0, 7, 12, 4]
            season_labels = ["No seasonality", "Weekly (7)", "Monthly (12)", "Quarterly (4)"]
            
            # Safely get the index for default_season
            try:
                season_index = season_options.index(default_season) if default_season in season_options else 2
            except (ValueError, AttributeError):
                season_index = 2
                default_season = 12  # Default to monthly seasonality if invalid
            
            season = st.selectbox("Seasonal period", season_options, 
                                index=season_index,
                                format_func=lambda x: season_labels[season_options.index(x)],
                                help="0 = no seasonality; 7 = weekly; 12 = monthly; 4 = quarterly",
                                key=f"ts_season_{forecast_form_key}")
        
        if st.button("Build forecast"):
            # Keep expander open when building forecast
            st.session_state['forecast_expander_open'] = True
            if not time_col or not target_col:
                st.warning("Select both time and target columns.")
            else:
                try:
                    # Validate that columns exist
                    if time_col not in dfb.columns:
                        st.error(f"Time column '{time_col}' not found in the dataset.")
                        st.stop()
                    if target_col not in dfb.columns:
                        st.error(f"Target column '{target_col}' not found in the dataset.")
                        st.stop()
                    
                    # Create time series DataFrame
                    df_ts = dfb[[time_col, target_col]].dropna().copy()
                    
                    # Check if we have enough data
                    if len(df_ts) < 3:
                        st.error("Not enough data points for forecasting. Need at least 3 observations.")
                        st.stop()
                    
                    # Parse dates
                    try:
                        df_ts[time_col] = pd.to_datetime(df_ts[time_col], errors="coerce")
                    except Exception as e:
                        log_error(e, f"Date parsing failed for column: {time_col}")
                        st.error("Could not parse the time column as dates. Please ensure the column contains valid date values.")
                        st.stop()
                    
                    # Remove rows with invalid dates
                    df_ts = df_ts.dropna(subset=[time_col])
                    
                    if len(df_ts) < 3:
                        st.error("Not enough valid data points after date parsing. Need at least 3 observations.")
                        st.stop()
                    
                    # Group by time frequency
                    df_ts = df_ts.groupby(pd.Grouper(key=time_col, freq=freq))[target_col].sum().reset_index()
                    df_ts = df_ts.sort_values(by=time_col)
                    df_ts = df_ts.set_index(time_col)
                    
                    # Create time series
                    series = df_ts[target_col].asfreq(freq)
                    
                    # Check if series has enough data
                    if len(series.dropna()) < 3:
                        st.error("Not enough data points after frequency aggregation. Try a different frequency or check your data.")
                        st.stop()
                    
                except Exception as e:
                    log_error(e, f"Data preparation failed for forecasting")
                    st.error("Failed to prepare data for forecasting. Please check your column selections and data quality.")
                    st.stop()

                try:
                    if selected_model == "Holt-Winters":
                        if season == 0:
                            model = ExponentialSmoothing(series, trend="add", seasonal=None)
                        else:
                            model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=season)
                        fit = model.fit(optimized=True)
                        forecast = fit.forecast(horizon)
                        
                        # Plot with safe DataFrame creation
                        try:
                            hist_df = series.reset_index().rename(columns={time_col: "Date", target_col: "Value"})
                            
                            # Safely create forecast DataFrame
                            if hasattr(forecast, 'reset_index'):
                                fc_df = forecast.reset_index()
                                if len(fc_df.columns) >= 2:
                                    fc_df.columns = ["Date", "Value"]
                                else:
                                    # Handle case where forecast is a Series
                                    fc_df = pd.DataFrame({
                                        'Date': pd.date_range(series.index[-1], periods=horizon+1, freq=freq)[1:],
                                        'Value': forecast.values
                                    })
                            else:
                                # Handle case where forecast is not a DataFrame
                                fc_df = pd.DataFrame({
                                    'Date': pd.date_range(series.index[-1], periods=horizon+1, freq=freq)[1:],
                                    'Value': forecast
                                })
                        except Exception as e:
                            log_error(e, f"Forecast DataFrame creation failed for Holt-Winters")
                            st.error("Failed to create forecast visualization. Please try again.")
                            st.stop()

                        fig_f = px.line(hist_df, x="Date", y="Value", title=f"Holt-Winters Forecast for {target_col}")
                        fig_f.add_scatter(x=fc_df["Date"], y=fc_df["Value"], mode="lines", name="Forecast")
                        st.plotly_chart(fig_f, use_container_width=True)

                        with st.expander("Forecast values"):
                            st.dataframe(fc_df, use_container_width=True)
                        
                        # AI-generated forecast summary and insights
                        if client is not None and st.toggle("🤖 Generate AI Forecast Analysis", value=True, key="holt_winters_ai_analysis"):
                            try:
                                # Prepare data for AI analysis
                                historical_data = hist_df.head(20).to_csv(index=False)
                                forecast_data = fc_df.to_csv(index=False)
                                
                                prompt = f"""
                                Analyze this Holt-Winters forecast and provide comprehensive insights.
                                
                                Forecast Details:
                                - Model: Holt-Winters Exponential Smoothing
                                - Target Variable: {target_col}
                                - Time Column: {time_col}
                                - Forecast Horizon: {horizon} periods
                                - Frequency: {freq}
                                - Seasonal Period: {season}
                                
                                Historical Data (last 20 periods):
                                {historical_data}
                                
                                Forecast Data:
                                {forecast_data}
                                
                                Please provide:
                                1. **Trend Analysis**: What trends are visible in the historical data?
                                2. **Seasonality**: Is there evidence of seasonal patterns?
                                3. **Forecast Interpretation**: What does the forecast predict?
                                4. **Confidence Assessment**: How reliable is this forecast?
                                5. **Business Implications**: What actions should be taken based on this forecast?
                                6. **Risk Factors**: What could affect forecast accuracy?
                                7. **Recommendations**: Specific next steps for stakeholders
                                
                                Keep it concise, actionable, and business-focused.
                                """
                                
                                resp = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "You are a senior data scientist specializing in time series forecasting. Provide clear, actionable insights from forecast results."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    temperature=0.3,
                                    max_tokens=600
                                )
                                
                                st.subheader("🤖 AI Forecast Analysis")
                                st.write(resp.choices[0].message.content)
                                
                            except Exception as e:
                                log_error(e, f"AI Holt-Winters forecast analysis failed")
                                st.warning("AI analysis failed. The forecast chart is still available.")
                    
                    elif selected_model == "Prophet":
                        if PROPHET_AVAILABLE:
                            prophet_model, prophet_forecast = prophet_forecast(df_ts.reset_index(), time_col, target_col, horizon)
                            if prophet_forecast is not None:
                                # Plot
                                hist_df = series.reset_index().rename(columns={time_col: "Date", target_col: "Value"})
                                fc_df = prophet_forecast[['ds', 'yhat']].tail(horizon).rename(columns={'ds': 'Date', 'yhat': 'Value'})

                                fig_f = px.line(hist_df, x="Date", y="Value", title=f"Prophet Forecast for {target_col}")
                                fig_f.add_scatter(x=fc_df["Date"], y=fc_df["Value"], mode="lines", name="Forecast")
                                st.plotly_chart(fig_f, use_container_width=True)

                                with st.expander("Forecast values"):
                                    st.dataframe(fc_df, use_container_width=True)
                                
                                # AI-generated forecast summary and insights for Prophet
                                if client is not None and st.toggle("🤖 Generate AI Forecast Analysis", value=True, key="prophet_ai_analysis"):
                                    try:
                                        # Prepare data for AI analysis
                                        historical_data = hist_df.head(20).to_csv(index=False)
                                        forecast_data = fc_df.to_csv(index=False)
                                        
                                        prompt = f"""
                                        Analyze this Prophet forecast and provide comprehensive insights.
                                        
                                        Forecast Details:
                                        - Model: Facebook Prophet
                                        - Target Variable: {target_col}
                                        - Time Column: {time_col}
                                        - Forecast Horizon: {horizon} periods
                                        - Frequency: {freq}
                                        
                                        Historical Data (last 20 periods):
                                        {historical_data}
                                        
                                        Forecast Data:
                                        {forecast_data}
                                        
                                        Please provide:
                                        1. **Trend Analysis**: What trends are visible in the historical data?
                                        2. **Seasonality**: Is there evidence of seasonal patterns?
                                        3. **Forecast Interpretation**: What does the forecast predict?
                                        4. **Confidence Assessment**: How reliable is this forecast?
                                        5. **Business Implications**: What actions should be taken based on this forecast?
                                        6. **Risk Factors**: What could affect forecast accuracy?
                                        7. **Recommendations**: Specific next steps for stakeholders
                                        
                                        Keep it concise, actionable, and business-focused.
                                        """
                                        
                                        resp = client.chat.completions.create(
                                            model="gpt-4o-mini",
                                            messages=[
                                                {"role": "system", "content": "You are a senior data scientist specializing in time series forecasting. Provide clear, actionable insights from forecast results."},
                                                {"role": "user", "content": prompt}
                                            ],
                                            temperature=0.3,
                                            max_tokens=600
                                        )
                                        
                                        st.subheader("🤖 AI Forecast Analysis")
                                        st.write(resp.choices[0].message.content)
                                        
                                    except Exception as e:
                                        log_error(e, f"AI Prophet forecast analysis failed")
                                        st.warning("AI analysis failed. The forecast chart is still available.")
                        else:
                            st.error("Prophet not available")
                    
                    elif selected_model == "ARIMA":
                        arima_model, arima_forecast = arima_forecast(df_ts.reset_index(), time_col, target_col, horizon)
                        if arima_forecast is not None:
                            # Plot
                            hist_df = series.reset_index().rename(columns={time_col: "Date", target_col: "Value"})
                            fc_df = pd.DataFrame({
                                'Date': pd.date_range(series.index[-1], periods=horizon+1, freq=freq)[1:],
                                'Value': arima_forecast.values
                            })

                            fig_f = px.line(hist_df, x="Date", y="Value", title=f"ARIMA Forecast for {target_col}")
                            fig_f.add_scatter(x=fc_df["Date"], y=fc_df["Value"], mode="lines", name="Forecast")
                            st.plotly_chart(fig_f, use_container_width=True)

                            with st.expander("Forecast values"):
                                st.dataframe(fc_df, use_container_width=True)
                            
                            # AI-generated forecast summary and insights for ARIMA
                            if client is not None and st.toggle("🤖 Generate AI Forecast Analysis", value=True, key="arima_ai_analysis"):
                                try:
                                    # Prepare data for AI analysis
                                    historical_data = hist_df.head(20).to_csv(index=False)
                                    forecast_data = fc_df.to_csv(index=False)
                                    
                                    prompt = f"""
                                    Analyze this ARIMA forecast and provide comprehensive insights.
                                    
                                    Forecast Details:
                                    - Model: ARIMA (AutoRegressive Integrated Moving Average)
                                    - Target Variable: {target_col}
                                    - Time Column: {time_col}
                                    - Forecast Horizon: {horizon} periods
                                    - Frequency: {freq}
                                    
                                    Historical Data (last 20 periods):
                                    {historical_data}
                                    
                                    Forecast Data:
                                    {forecast_data}
                                    
                                    Please provide:
                                    1. **Trend Analysis**: What trends are visible in the historical data?
                                    2. **Seasonality**: Is there evidence of seasonal patterns?
                                    3. **Forecast Interpretation**: What does the forecast predict?
                                    4. **Confidence Assessment**: How reliable is this forecast?
                                    5. **Business Implications**: What actions should be taken based on this forecast?
                                    6. **Risk Factors**: What could affect forecast accuracy?
                                    7. **Recommendations**: Specific next steps for stakeholders
                                    
                                    Keep it concise, actionable, and business-focused.
                                    """
                                    
                                    resp = client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=[
                                            {"role": "system", "content": "You are a senior data scientist specializing in time series forecasting. Provide clear, actionable insights from forecast results."},
                                            {"role": "user", "content": prompt}
                                        ],
                                        temperature=0.3,
                                        max_tokens=600
                                    )
                                    
                                    st.subheader("🤖 AI Forecast Analysis")
                                    st.write(resp.choices[0].message.content)
                                    
                                except Exception as e:
                                    log_error(e, f"AI forecast analysis failed for {selected_model}")
                                    st.warning("AI analysis failed. The forecast chart is still available.")
                    
                    elif selected_model == "Ensemble":
                        ensemble_forecasts = ensemble_forecast(df_ts.reset_index(), time_col, target_col, horizon)
                        if ensemble_forecasts is not None:
                            # Plot
                            hist_df = series.reset_index().rename(columns={time_col: "Date", target_col: "Value"})
                            
                            fig_f = px.line(hist_df, x="Date", y="Value", title=f"Ensemble Forecast for {target_col}")
                            
                            # Add each model's forecast
                            for model_name, forecast_values in ensemble_forecasts.items():
                                if model_name != "Ensemble":
                                    fc_df = pd.DataFrame({
                                        'Date': pd.date_range(series.index[-1], periods=horizon+1, freq=freq)[1:],
                                        'Value': forecast_values
                                    })
                                    fig_f.add_scatter(x=fc_df["Date"], y=fc_df["Value"], mode="lines", name=f"{model_name}")
                            
                            # Add ensemble forecast
                            ensemble_fc_df = pd.DataFrame({
                                'Date': pd.date_range(series.index[-1], periods=horizon+1, freq=freq)[1:],
                                'Value': ensemble_forecasts['Ensemble']
                            })
                            fig_f.add_scatter(x=ensemble_fc_df["Date"], y=ensemble_fc_df["Value"], 
                                            mode="lines", name="Ensemble", line=dict(width=4))
                            
                            st.plotly_chart(fig_f, use_container_width=True)

                            with st.expander("Ensemble forecast values"):
                                st.dataframe(ensemble_fc_df, use_container_width=True)
                            
                            with st.expander("Model comparison"):
                                comparison_df = pd.DataFrame(ensemble_forecasts)
                                st.dataframe(comparison_df, use_container_width=True)
                            
                            # AI-generated forecast summary and insights for Ensemble
                            if client is not None and st.toggle("🤖 Generate AI Forecast Analysis", value=True, key="ensemble_ai_analysis"):
                                try:
                                    # Prepare data for AI analysis
                                    historical_data = hist_df.head(20).to_csv(index=False)
                                    ensemble_forecast_data = ensemble_fc_df.to_csv(index=False)
                                    comparison_data = comparison_df.to_csv(index=False)
                                    
                                    prompt = f"""
                                    Analyze this Ensemble forecast and provide comprehensive insights.
                                    
                                    Forecast Details:
                                    - Model: Ensemble (Combining Holt-Winters, Prophet, and ARIMA)
                                    - Target Variable: {target_col}
                                    - Time Column: {time_col}
                                    - Forecast Horizon: {horizon} periods
                                    - Frequency: {freq}
                                    
                                    Historical Data (last 20 periods):
                                    {historical_data}
                                    
                                    Ensemble Forecast Data:
                                    {ensemble_forecast_data}
                                    
                                    Model Comparison Data:
                                    {comparison_data}
                                    
                                    Please provide:
                                    1. **Trend Analysis**: What trends are visible in the historical data?
                                    2. **Seasonality**: Is there evidence of seasonal patterns?
                                    3. **Forecast Interpretation**: What does the ensemble forecast predict?
                                    4. **Model Agreement**: How do the individual models compare?
                                    5. **Confidence Assessment**: How reliable is this ensemble forecast?
                                    6. **Business Implications**: What actions should be taken based on this forecast?
                                    7. **Risk Factors**: What could affect forecast accuracy?
                                    8. **Recommendations**: Specific next steps for stakeholders
                                    
                                    Keep it concise, actionable, and business-focused.
                                    """
                                    
                                    resp = client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=[
                                            {"role": "system", "content": "You are a senior data scientist specializing in time series forecasting. Provide clear, actionable insights from ensemble forecast results."},
                                            {"role": "user", "content": prompt}
                                        ],
                                        temperature=0.3,
                                        max_tokens=700
                                    )
                                    
                                    st.subheader("🤖 AI Forecast Analysis")
                                    st.write(resp.choices[0].message.content)
                                    
                                except Exception as e:
                                    log_error(e, f"AI ensemble forecast analysis failed")
                                    st.warning("AI analysis failed. The forecast chart is still available.")
                
                except Exception as e:
                    log_error(e, f"Forecasting failed for model: {selected_model}, target: {target_col}")
                    st.error("Forecast generation failed. Please check your data and try again with different parameters.")
        
        # Add a reset button at the bottom of the expander
        st.divider()
        if st.button("🔄 Reset Forecast Form", key="reset_forecast_form"):
            st.session_state['forecast_expander_open'] = False
            st.session_state['forecast_form_key'] = st.session_state.get('forecast_form_key', 0) + 1

        # =========================
        # Download
        # =========================
        st.header("📥 Download")
        
        # Check if we have enhanced data (with AI-generated columns)
        enhanced_data_available = 'enhanced_data' in st.session_state and st.session_state['enhanced_data'] is not None
        
        if enhanced_data_available:
            st.info("🎉 **Enhanced dataset with AI-generated data is available!**")
            enhanced_data = st.session_state['enhanced_data']
            enhanced_columns = st.session_state.get('enhanced_columns', [])
            
            st.write(f"**AI-generated columns:** {', '.join(enhanced_columns)}")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.download_button("Download Original CSV", data=df_to_download_bytes(dfb, "csv"), file_name="original_data.csv", mime="text/csv")
            with c2:
                st.download_button("Download Original Excel", data=df_to_download_bytes(dfb, "xlsx"), file_name="original_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with c3:
                st.download_button("Download Enhanced CSV", data=df_to_download_bytes(enhanced_data, "csv"), file_name="enhanced_data.csv", mime="text/csv")
            with c4:
                st.download_button("Download Enhanced Excel", data=df_to_download_bytes(enhanced_data, "xlsx"), file_name="enhanced_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Download CSV", data=df_to_download_bytes(dfb, "csv"), file_name="beautified_data.csv", mime="text/csv")
            with c2:
                                 st.download_button("Download Excel", data=df_to_download_bytes(dfb, "xlsx"), file_name="beautified_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    if merged is None:
        st.info("📁 Upload files or try the sample dataset to get started.")
    else:
        st.info("Upload one or more CSV/Excel files to get started.")
