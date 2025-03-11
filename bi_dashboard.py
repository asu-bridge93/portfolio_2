import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from datetime import datetime, timedelta
import io

# Set page configuration
st.set_page_config(
    page_title="Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 600;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
    .trend-up {
        color: #4CAF50;
    }
    .trend-down {
        color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data storage
if 'data' not in st.session_state:
    st.session_state.data = None
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None

# Sample data generation function
def generate_sample_data():
    # Generate dates for the past 2 years
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(730, 0, -1)]
    
    # Generate sales data with seasonal patterns and trend
    base_sales = 1000
    trend = np.linspace(0, 500, len(dates))
    seasonal = 200 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    weekly = 100 * np.sin(np.linspace(0, 52*2*np.pi, len(dates)))
    noise = np.random.normal(0, 50, len(dates))
    
    sales = base_sales + trend + seasonal + weekly + noise
    sales = np.maximum(sales, 0)  # Ensure no negative sales
    
    # Generate product categories and regions
    categories = ['Electronics', 'Clothing', 'Home Goods', 'Food', 'Beauty']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Create dataframe
    data = []
    for i, date in enumerate(dates):
        for category in categories:
            for region in regions:
                # Add variation by category and region
                cat_factor = 0.7 + 0.6 * categories.index(category) / len(categories)
                reg_factor = 0.8 + 0.4 * regions.index(region) / len(regions)
                
                # Calculate sales for this combination
                daily_sales = sales[i] * cat_factor * reg_factor
                
                # Add some randomness
                daily_sales *= np.random.uniform(0.9, 1.1)
                
                # Calculate profit (30-50% of sales depending on category)
                profit_margin = 0.3 + 0.2 * categories.index(category) / len(categories)
                profit = daily_sales * profit_margin
                
                # Calculate units sold (sales / average price)
                avg_price = 50 + 50 * categories.index(category) / len(categories)
                units = int(daily_sales / avg_price)
                
                data.append({
                    'Date': date,
                    'Category': category,
                    'Region': region,
                    'Sales': round(daily_sales, 2),
                    'Profit': round(profit, 2),
                    'Units': units,
                    'Customers': int(units * np.random.uniform(0.8, 1.2))
                })
    
    return pd.DataFrame(data)

# Sidebar navigation
def sidebar():
    with st.sidebar:
        st.image("https://placeholder.svg?height=100&width=300", width=200)
        st.title("BI Dashboard")
        
        # Navigation
        st.subheader("Navigation")
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.page = 'dashboard'
        if st.button("📈 Sales Analysis", use_container_width=True):
            st.session_state.page = 'sales'
        if st.button("🔮 Forecasting", use_container_width=True):
            st.session_state.page = 'forecast'
        if st.button("🧩 Segment Analysis", use_container_width=True):
            st.session_state.page = 'segment'
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.page = 'settings'
        
        # Data upload section
        st.subheader("Data")
        uploaded_file = st.file_uploader("Upload your data (CSV)", type="csv")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error: {e}")
        
        if st.button("Use Sample Data"):
            with st.spinner("Generating sample data..."):
                st.session_state.data = generate_sample_data()
                st.success("Sample data loaded!")
        
        if st.session_state.data is not None:
            st.info(f"Loaded data: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
        
        st.markdown("---")
        st.caption("© 2025 BI Dashboard")

# Dashboard page
def dashboard_page():
    st.markdown("<h1 class='main-header'>Business Intelligence Dashboard</h1>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.info("Please upload data or use sample data from the sidebar to get started.")
        return
    
    # Get the data
    df = st.session_state.data
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        date_range = st.date_input(
            "Date Range",
            [df['Date'].min().date(), df['Date'].max().date()],
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date()
        )
    with col2:
        selected_categories = st.multiselect(
            "Categories",
            options=df['Category'].unique(),
            default=df['Category'].unique()
        )
    with col3:
        selected_regions = st.multiselect(
            "Regions",
            options=df['Region'].unique(),
            default=df['Region'].unique()
        )
    
    # Apply filters
    filtered_df = df[
        (df['Date'].dt.date >= date_range[0]) &
        (df['Date'].dt.date <= date_range[1]) &
        (df['Category'].isin(selected_categories)) &
        (df['Region'].isin(selected_regions))
    ]
    
    # Calculate KPIs
    total_sales = filtered_df['Sales'].sum()
    total_profit = filtered_df['Profit'].sum()
    total_units = filtered_df['Units'].sum()
    total_customers = filtered_df['Customers'].sum()
    
    # Calculate previous period for comparison
    date_diff = (date_range[1] - date_range[0]).days
    prev_end_date = date_range[0] - timedelta(days=1)
    prev_start_date = prev_end_date - timedelta(days=date_diff)
    
    prev_filtered_df = df[
        (df['Date'].dt.date >= prev_start_date) &
        (df['Date'].dt.date <= prev_end_date) &
        (df['Category'].isin(selected_categories)) &
        (df['Region'].isin(selected_regions))
    ]
    
    prev_sales = prev_filtered_df['Sales'].sum()
    prev_profit = prev_filtered_df['Profit'].sum()
    prev_units = prev_filtered_df['Units'].sum()
    prev_customers = prev_filtered_df['Customers'].sum()
    
    # Calculate percentage changes
    sales_change = ((total_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0
    profit_change = ((total_profit - prev_profit) / prev_profit * 100) if prev_profit > 0 else 0
    units_change = ((total_units - prev_units) / prev_units * 100) if prev_units > 0 else 0
    customers_change = ((total_customers - prev_customers) / prev_customers * 100) if prev_customers > 0 else 0
    
    # Display KPI cards
    st.markdown("<h2 class='sub-header'>Key Performance Indicators</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Total Sales</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>${total_sales:,.2f}</p>", unsafe_allow_html=True)
        trend_class = "trend-up" if sales_change >= 0 else "trend-down"
        trend_icon = "↑" if sales_change >= 0 else "↓"
        st.markdown(f"<p class='{trend_class}'>{trend_icon} {abs(sales_change):.1f}% vs previous period</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Total Profit</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>${total_profit:,.2f}</p>", unsafe_allow_html=True)
        trend_class = "trend-up" if profit_change >= 0 else "trend-down"
        trend_icon = "↑" if profit_change >= 0 else "↓"
        st.markdown(f"<p class='{trend_class}'>{trend_icon} {abs(profit_change):.1f}% vs previous period</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Units Sold</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{total_units:,}</p>", unsafe_allow_html=True)
        trend_class = "trend-up" if units_change >= 0 else "trend-down"
        trend_icon = "↑" if units_change >= 0 else "↓"
        st.markdown(f"<p class='{trend_class}'>{trend_icon} {abs(units_change):.1f}% vs previous period</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Total Customers</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{total_customers:,}</p>", unsafe_allow_html=True)
        trend_class = "trend-up" if customers_change >= 0 else "trend-down"
        trend_icon = "↑" if customers_change >= 0 else "↓"
        st.markdown(f"<p class='{trend_class}'>{trend_icon} {abs(customers_change):.1f}% vs previous period</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Time series chart
    st.markdown("<h2 class='sub-header'>Sales Trend</h2>", unsafe_allow_html=True)
    
    # Aggregate data by date
    daily_data = filtered_df.groupby(filtered_df['Date'].dt.date).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Units': 'sum',
        'Customers': 'sum'
    }).reset_index()
    
    # Create time series chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=daily_data['Date'],
            y=daily_data['Sales'],
            name="Sales",
            line=dict(color='#1E88E5', width=3)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_data['Date'],
            y=daily_data['Profit'],
            name="Profit",
            line=dict(color='#43A047', width=3, dash='dot')
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            x=daily_data['Date'],
            y=daily_data['Units'],
            name="Units",
            marker_color='rgba(156, 39, 176, 0.3)'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Sales, Profit and Units Over Time",
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Amount ($)", secondary_y=False)
    fig.update_yaxes(title_text="Units", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Category and Region Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h2 class='sub-header'>Sales by Category</h2>", unsafe_allow_html=True)
        category_data = filtered_df.groupby('Category').agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).reset_index()
        
        category_data['Profit Margin'] = category_data['Profit'] / category_data['Sales'] * 100
        category_data = category_data.sort_values('Sales', ascending=False)
        
        fig = px.bar(
            category_data,
            x='Category',
            y='Sales',
            color='Profit Margin',
            color_continuous_scale='Viridis',
            text=category_data['Sales'].apply(lambda x: f"${x:,.0f}")
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Sales ($)",
            coloraxis_colorbar_title="Profit Margin (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<h2 class='sub-header'>Sales by Region</h2>", unsafe_allow_html=True)
        region_data = filtered_df.groupby('Region').agg({
            'Sales': 'sum'
        }).reset_index()
        
        region_data = region_data.sort_values('Sales', ascending=False)
        
        fig = px.pie(
            region_data,
            values='Sales',
            names='Region',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Sales: $%{value:,.2f}<br>Percentage: %{percent}'
        )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap of Sales by Category and Region
    st.markdown("<h2 class='sub-header'>Sales Heatmap: Category vs Region</h2>", unsafe_allow_html=True)
    
    heatmap_data = filtered_df.pivot_table(
        index='Category',
        columns='Region',
        values='Sales',
        aggfunc='sum'
    )
    
    fig = px.imshow(
        heatmap_data,
        text_auto='.2s',
        aspect="auto",
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Category",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Sales Analysis page
def sales_analysis_page():
    st.markdown("<h1 class='main-header'>Sales Analysis</h1>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.info("Please upload data or use sample data from the sidebar to get started.")
        return
    
    # Get the data
    df = st.session_state.data
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        date_range = st.date_input(
            "Date Range",
            [df['Date'].min().date(), df['Date'].max().date()],
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date()
        )
    with col2:
        selected_categories = st.multiselect(
            "Categories",
            options=df['Category'].unique(),
            default=df['Category'].unique()
        )
    with col3:
        selected_regions = st.multiselect(
            "Regions",
            options=df['Region'].unique(),
            default=df['Region'].unique()
        )
    
    # Apply filters
    filtered_df = df[
        (df['Date'].dt.date >= date_range[0]) &
        (df['Date'].dt.date <= date_range[1]) &
        (df['Category'].isin(selected_categories)) &
        (df['Region'].isin(selected_regions))
    ]
    
    # Time period analysis
    st.markdown("<h2 class='sub-header'>Time Period Analysis</h2>", unsafe_allow_html=True)
    
    period_options = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
    period = st.selectbox("Select Time Period", period_options)
    
    if period == "Daily":
        filtered_df['Period'] = filtered_df['Date'].dt.date
    elif period == "Weekly":
        filtered_df['Period'] = filtered_df['Date'].dt.to_period('W').apply(lambda r: r.start_time)
    elif period == "Monthly":
        filtered_df['Period'] = filtered_df['Date'].dt.to_period('M').apply(lambda r: r.start_time)
    elif period == "Quarterly":
        filtered_df['Period'] = filtered_df['Date'].dt.to_period('Q').apply(lambda r: r.start_time)
    else:  # Yearly
        filtered_df['Period'] = filtered_df['Date'].dt.to_period('Y').apply(lambda r: r.start_time)
    
    period_data = filtered_df.groupby('Period').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Units': 'sum',
        'Customers': 'sum'
    }).reset_index()
    
    # Calculate additional metrics
    period_data['Profit Margin'] = period_data['Profit'] / period_data['Sales'] * 100
    period_data['Avg Sale Value'] = period_data['Sales'] / period_data['Customers']
    period_data['Units per Customer'] = period_data['Units'] / period_data['Customers']
    
    # Create time series chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=period_data['Period'],
            y=period_data['Sales'],
            name="Sales",
            line=dict(color='#1E88E5', width=3)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=period_data['Period'],
            y=period_data['Profit'],
            name="Profit",
            line=dict(color='#43A047', width=3, dash='dot')
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=period_data['Period'],
            y=period_data['Profit Margin'],
            name="Profit Margin (%)",
            line=dict(color='#FFA000', width=2)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f"{period} Sales and Profit Analysis",
        xaxis_title="Period",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Amount ($)", secondary_y=False)
    fig.update_yaxes(title_text="Profit Margin (%)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sales Metrics Analysis
    st.markdown("<h2 class='sub-header'>Sales Metrics Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        metric_options = ["Avg Sale Value", "Units per Customer", "Profit Margin"]
        selected_metric = st.selectbox("Select Metric", metric_options)
        
        fig = px.line(
            period_data,
            x='Period',
            y=selected_metric,
            markers=True,
            line_shape='spline',
            color_discrete_sequence=['#5E35B1']
        )
        
        fig.update_layout(
            title=f"{selected_metric} Over Time",
            xaxis_title="Period",
            yaxis_title=selected_metric,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Correlation analysis
        correlation_df = period_data[['Sales', 'Profit', 'Units', 'Customers', 'Profit Margin', 'Avg Sale Value']]
        correlation_matrix = correlation_df.corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )
        
        fig.update_layout(
            title="Correlation Matrix of Key Metrics",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Performers
    st.markdown("<h2 class='sub-header'>Top Performers</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Top Categories</h3>", unsafe_allow_html=True)
        
        category_data = filtered_df.groupby('Category').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Units': 'sum'
        }).reset_index()
        
        category_data['Profit Margin'] = category_data['Profit'] / category_data['Sales'] * 100
        category_data = category_data.sort_values('Sales', ascending=False)
        
        fig = px.bar(
            category_data,
            x='Category',
            y=['Sales', 'Profit'],
            barmode='group',
            color_discrete_sequence=['#1E88E5', '#43A047']
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Amount ($)",
            legend_title="Metric",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            category_data[['Category', 'Sales', 'Profit', 'Profit Margin', 'Units']].sort_values('Sales', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("<h3>Top Regions</h3>", unsafe_allow_html=True)
        
        region_data = filtered_df.groupby('Region').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Units': 'sum'
        }).reset_index()
        
        region_data['Profit Margin'] = region_data['Profit'] / region_data['Sales'] * 100
        region_data = region_data.sort_values('Sales', ascending=False)
        
        fig = px.bar(
            region_data,
            x='Region',
            y=['Sales', 'Profit'],
            barmode='group',
            color_discrete_sequence=['#1E88E5', '#43A047']
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Amount ($)",
            legend_title="Metric",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            region_data[['Region', 'Sales', 'Profit', 'Profit Margin', 'Units']].sort_values('Sales', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    
    # Category-Region Combination Analysis
    st.markdown("<h2 class='sub-header'>Category-Region Combination Analysis</h2>", unsafe_allow_html=True)
    
    combo_data = filtered_df.groupby(['Category', 'Region']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Units': 'sum'
    }).reset_index()
    
    combo_data['Profit Margin'] = combo_data['Profit'] / combo_data['Sales'] * 100
    combo_data = combo_data.sort_values('Sales', ascending=False).head(10)
    
    fig = px.bar(
        combo_data,
        x='Sales',
        y='Category',
        color='Region',
        orientation='h',
        text=combo_data['Sales'].apply(lambda x: f"${x:,.0f}"),
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        title="Top 10 Category-Region Combinations by Sales",
        xaxis_title="Sales ($)",
        yaxis_title="",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table with all combinations
    st.markdown("<h3>All Category-Region Combinations</h3>", unsafe_allow_html=True)
    
    combo_data_all = filtered_df.groupby(['Category', 'Region']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Units': 'sum'
    }).reset_index()
    
    combo_data_all['Profit Margin'] = combo_data_all['Profit'] / combo_data_all['Sales'] * 100
    combo_data_all = combo_data_all.sort_values('Sales', ascending=False)
    
    st.dataframe(
        combo_data_all,
        use_container_width=True,
        hide_index=True
    )

# Forecasting page
def forecasting_page():
    st.markdown("<h1 class='main-header'>Sales Forecasting</h1>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.info("Please upload data or use sample data from the sidebar to get started.")
        return
    
    # Get the data
    df = st.session_state.data
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        forecast_category = st.selectbox(
            "Select Category for Forecast",
            options=['All Categories'] + list(df['Category'].unique())
        )
    with col2:
        forecast_region = st.selectbox(
            "Select Region for Forecast",
            options=['All Regions'] + list(df['Region'].unique())
        )
    
    # Filter data based on selection
    if forecast_category == 'All Categories' and forecast_region == 'All Regions':
        filtered_df = df
    elif forecast_category == 'All Categories':
        filtered_df = df[df['Region'] == forecast_region]
    elif forecast_region == 'All Regions':
        filtered_df = df[df['Category'] == forecast_category]
    else:
        filtered_df = df[(df['Category'] == forecast_category) & (df['Region'] == forecast_region)]
    
    # Aggregate data by date
    daily_data = filtered_df.groupby(filtered_df['Date'].dt.date).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Units': 'sum'
    }).reset_index()
    
    # Convert to time series
    daily_data.set_index('Date', inplace=True)
    
    # Forecast parameters
    st.markdown("<h2 class='sub-header'>Forecast Parameters</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30, step=1)
    with col2:
        forecast_metric = st.selectbox("Forecast Metric", options=['Sales', 'Profit', 'Units'])
    with col3:
        forecast_model = st.selectbox("Forecast Model", options=['Linear Regression', 'Polynomial Regression', 'SARIMA'])
    
    # Prepare data for forecasting
    ts_data = daily_data[forecast_metric].reset_index()
    ts_data.columns = ['ds', 'y']
    
    # Convert ds column to datetime explicitly to fix the error
    ts_data['ds'] = pd.to_datetime(ts_data['ds'])
    
    # Add time features
    ts_data['day_of_week'] = ts_data['ds'].dt.dayofweek
    ts_data['month'] = ts_data['ds'].dt.month
    ts_data['day'] = ts_data['ds'].dt.day
    
    # Create training data
    X = np.array(range(len(ts_data))).reshape(-1, 1)
    y = ts_data['y'].values
    
    # Generate future dates
    last_date = ts_data['ds'].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Add time features to future data
    future_df['day_of_week'] = future_df['ds'].dt.dayofweek
    future_df['month'] = future_df['ds'].dt.month
    future_df['day'] = future_df['ds'].dt.day
    
    # Create future X
    X_future = np.array(range(len(ts_data), len(ts_data) + len(future_df))).reshape(-1, 1)
    
    # Forecast based on selected model
    if forecast_model == 'Linear Regression':
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        y_future = model.predict(X_future)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        model_name = "Linear Regression"
    
    elif forecast_model == 'Polynomial Regression':
        degree = 2
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        X_future_poly = poly.transform(X_future)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Make predictions
        y_pred = model.predict(X_poly)
        y_future = model.predict(X_future_poly)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        model_name = f"Polynomial Regression (degree={degree})"
    
    else:  # SARIMA
        # Prepare data for SARIMA
        ts_series = pd.Series(y, index=pd.DatetimeIndex(ts_data['ds']))
        
        # Fit SARIMA model
        model = sm.tsa.statespace.SARIMAX(
            ts_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        results = model.fit(disp=False)
        
        # Make predictions
        y_pred = results.fittedvalues
        y_future = results.forecast(steps=forecast_days)
        
        # Calculate metrics
        mse = mean_squared_error(ts_series[1:], y_pred[1:])
        r2 = r2_score(ts_series[1:], y_pred[1:])
        
        model_name = "SARIMA(1,1,1)(1,1,1,7)"
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Date': future_df['ds'],
        forecast_metric: y_future
    })
    
    # Store forecast data in session state
    st.session_state.forecast_data = forecast_df
    
    # Display forecast results
    st.markdown("<h2 class='sub-header'>Forecast Results</h2>", unsafe_allow_html=True)
    
    # Display model metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    with col2:
        st.metric("R² Score", f"{r2:.2f}")
    
    # Plot historical data and forecast
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(
        go.Scatter(
            x=ts_data['ds'],
            y=ts_data['y'],
            name="Historical Data",
            line=dict(color='#1E88E5', width=2)
        )
    )
    
    # Fitted values
    fig.add_trace(
        go.Scatter(
            x=ts_data['ds'],
            y=y_pred,
            name="Fitted Values",
            line=dict(color='#FFA000', width=2, dash='dot')
        )
    )
    
    # Forecast
    fig.add_trace(
        go.Scatter(
            x=future_df['ds'],
            y=y_future,
            name="Forecast",
            line=dict(color='#E53935', width=3)
        )
    )
    
    # Add confidence intervals for SARIMA
    if forecast_model == 'SARIMA':
        pred_ci = results.get_forecast(steps=forecast_days).conf_int()
        fig.add_trace(
            go.Scatter(
                x=future_df['ds'],
                y=pred_ci.iloc[:, 0],
                fill=None,
                mode='lines',
                line=dict(color='rgba(229, 57, 53, 0.3)'),
                name="Lower CI"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_df['ds'],
                y=pred_ci.iloc[:, 1],
                fill='tonexty',
                mode='lines',
                line=dict(color='rgba(229, 57, 53, 0.3)'),
                name="Upper CI"
            )
        )
    
    fig.update_layout(
        title=f"{forecast_metric} Forecast using {model_name}",
        xaxis_title="Date",
        yaxis_title=forecast_metric,
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display forecast data
    st.markdown("<h3>Forecast Data</h3>", unsafe_allow_html=True)
    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Download forecast data
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="Download Forecast Data",
        data=csv,
        file_name=f"{forecast_metric}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Forecast Interpretation
    st.markdown("<h2 class='sub-header'>Forecast Interpretation</h2>", unsafe_allow_html=True)
    
    # Calculate key metrics
    total_forecast = forecast_df[forecast_metric].sum()
    avg_forecast = forecast_df[forecast_metric].mean()
    max_forecast = forecast_df[forecast_metric].max()
    min_forecast = forecast_df[forecast_metric].min()
    
    # Calculate historical metrics for comparison
    total_historical = ts_data['y'].sum()
    avg_historical = ts_data['y'].mean()
    
    # Calculate percentage change
    period_length = min(len(ts_data), forecast_days)
    recent_historical = ts_data['y'].iloc[-period_length:].sum()
    pct_change = (total_forecast - recent_historical) / recent_historical * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Total Forecast {forecast_metric}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>${total_forecast:,.2f}</p>" if forecast_metric != 'Units' else f"<p class='metric-value'>{int(total_forecast):,}</p>", unsafe_allow_html=True)
        trend_class = "trend-up" if pct_change >= 0 else "trend-down"
        trend_icon = "↑" if pct_change >= 0 else "↓"
        st.markdown(f"<p class='{trend_class}'>{trend_icon} {abs(pct_change):.1f}% vs previous {period_length} days</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Average Daily {forecast_metric}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>${avg_forecast:,.2f}</p>" if forecast_metric != 'Units' else f"<p class='metric-value'>{int(avg_forecast):,}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Maximum Daily {forecast_metric}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>${max_forecast:,.2f}</p>" if forecast_metric != 'Units' else f"<p class='metric-value'>{int(max_forecast):,}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Minimum Daily {forecast_metric}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>${min_forecast:,.2f}</p>" if forecast_metric != 'Units' else f"<p class='metric-value'>{int(min_forecast):,}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Forecast insights
    st.markdown("<h3>Forecast Insights</h3>", unsafe_allow_html=True)
    
    insights = []
    
    # Trend insight
    if pct_change > 10:
        insights.append(f"Strong growth expected: {forecast_metric} is projected to increase by {pct_change:.1f}% compared to the previous period.")
    elif pct_change > 0:
        insights.append(f"Moderate growth expected: {forecast_metric} is projected to increase by {pct_change:.1f}% compared to the previous period.")
    elif pct_change > -10:
        insights.append(f"Slight decline expected: {forecast_metric} is projected to decrease by {abs(pct_change):.1f}% compared to the previous period.")
    else:
        insights.append(f"Significant decline expected: {forecast_metric} is projected to decrease by {abs(pct_change):.1f}% compared to the previous period.")
    
    # Seasonality insight
    if forecast_model == 'SARIMA':
        insights.append("The forecast accounts for weekly seasonality patterns in the data.")
    
    # Volatility insight
    volatility = forecast_df[forecast_metric].std() / forecast_df[forecast_metric].mean() * 100
    if volatility > 20:
        insights.append(f"High volatility expected: Daily {forecast_metric} may vary significantly (coefficient of variation: {volatility:.1f}%).")
    else:
        insights.append(f"Stable performance expected: Daily {forecast_metric} should remain relatively consistent (coefficient of variation: {volatility:.1f}%).")
    
    # Display insights
    for insight in insights:
        st.markdown(f"- {insight}")

# Segment Analysis page
def segment_analysis_page():
    st.markdown("<h1 class='main-header'>Segment Analysis</h1>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.info("Please upload data or use sample data from the sidebar to get started.")
        return
    
    # Get the data
    df = st.session_state.data
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        date_range = st.date_input(
            "Date Range",
            [df['Date'].min().date(), df['Date'].max().date()],
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date()
        )
    with col2:
        primary_dimension = st.selectbox(
            "Primary Dimension",
            options=['Category', 'Region']
        )
    with col3:
        secondary_dimension = st.selectbox(
            "Secondary Dimension",
            options=['Region', 'Category'],
            index=1 if primary_dimension == 'Region' else 0
        )
    
    # Apply date filter
    filtered_df = df[
        (df['Date'].dt.date >= date_range[0]) &
        (df['Date'].dt.date <= date_range[1])
    ]
    
    # Segment Analysis
    st.markdown("<h2 class='sub-header'>Segment Performance</h2>", unsafe_allow_html=True)
    
    # Aggregate data by primary dimension
    segment_data = filtered_df.groupby(primary_dimension).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Units': 'sum',
        'Customers': 'sum'
    }).reset_index()
    
    # Calculate additional metrics
    segment_data['Profit Margin'] = segment_data['Profit'] / segment_data['Sales'] * 100
    segment_data['Avg Sale Value'] = segment_data['Sales'] / segment_data['Customers']
    segment_data['Units per Customer'] = segment_data['Units'] / segment_data['Customers']
    
    # Sort by sales
    segment_data = segment_data.sort_values('Sales', ascending=False)
    
    # Create bubble chart
    fig = px.scatter(
        segment_data,
        x='Profit Margin',
        y='Sales',
        size='Customers',
        color=primary_dimension,
        hover_name=primary_dimension,
        text=primary_dimension,
        log_y=True if segment_data['Sales'].max() / segment_data['Sales'].min() > 100 else False
    )
    
    fig.update_traces(
        textposition='top center',
        marker=dict(sizemode='area', sizeref=0.1)
    )
    
    fig.update_layout(
        title=f"Segment Performance: {primary_dimension}",
        xaxis_title="Profit Margin (%)",
        yaxis_title="Sales ($)",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display segment data
    st.dataframe(
        segment_data,
        use_container_width=True,
        hide_index=True
    )
    
    # Cross-segment Analysis
    st.markdown("<h2 class='sub-header'>Cross-segment Analysis</h2>", unsafe_allow_html=True)
    
    # Aggregate data by both dimensions
    cross_segment_data = filtered_df.groupby([primary_dimension, secondary_dimension]).agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    # Create heatmap
    cross_pivot = cross_segment_data.pivot_table(
        index=primary_dimension,
        columns=secondary_dimension,
        values='Sales'
    )
    
    fig = px.imshow(
        cross_pivot,
        text_auto='.2s',
        aspect="auto",
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title=f"Sales Heatmap: {primary_dimension} vs {secondary_dimension}",
        xaxis_title=secondary_dimension,
        yaxis_title=primary_dimension,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment Contribution Analysis
    st.markdown("<h2 class='sub-header'>Segment Contribution Analysis</h2>", unsafe_allow_html=True)
    
    # Calculate total sales
    total_sales = filtered_df['Sales'].sum()
    
    # Calculate contribution by primary dimension
    contribution_data = segment_data[[primary_dimension, 'Sales']].copy()
    contribution_data['Contribution (%)'] = contribution_data['Sales'] / total_sales * 100
    contribution_data = contribution_data.sort_values('Contribution (%)', ascending=False)
    
    # Create pareto chart
    contribution_data['Cumulative (%)'] = contribution_data['Contribution (%)'].cumsum()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=contribution_data[primary_dimension],
            y=contribution_data['Contribution (%)'],
            name="Contribution (%)",
            marker_color='#1E88E5'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=contribution_data[primary_dimension],
            y=contribution_data['Cumulative (%)'],
            name="Cumulative (%)",
            line=dict(color='#E53935', width=3)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f"Pareto Analysis: {primary_dimension} Contribution to Total Sales",
        xaxis_title=primary_dimension,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    fig.update_yaxes(title_text="Contribution (%)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment Growth Analysis
    st.markdown("<h2 class='sub-header'>Segment Growth Analysis</h2>", unsafe_allow_html=True)
    
    # Calculate midpoint of date range
    mid_date = date_range[0] + (date_range[1] - date_range[0]) / 2
    
    # Split data into two periods
    period1_df = filtered_df[filtered_df['Date'].dt.date <= mid_date]
    period2_df = filtered_df[filtered_df['Date'].dt.date > mid_date]
    
    # Aggregate data by primary dimension for both periods
    period1_data = period1_df.groupby(primary_dimension).agg({
        'Sales': 'sum'
    }).reset_index()
    
    period2_data = period2_df.groupby(primary_dimension).agg({
        'Sales': 'sum'
    }).reset_index()
    
    # Merge data
    period1_data.columns = [primary_dimension, 'Period1_Sales']
    period2_data.columns = [primary_dimension, 'Period2_Sales']
    
    growth_data = pd.merge(period1_data, period2_data, on=primary_dimension)
    
    # Calculate growth
    growth_data['Growth (%)'] = (growth_data['Period2_Sales'] - growth_data['Period1_Sales']) / growth_data['Period1_Sales'] * 100
    growth_data['Absolute Growth'] = growth_data['Period2_Sales'] - growth_data['Period1_Sales']
    
    # Sort by growth
    growth_data = growth_data.sort_values('Growth (%)', ascending=False)
    
    # Create growth chart
    fig = px.bar(
        growth_data,
        x=primary_dimension,
        y='Growth (%)',
        color='Growth (%)',
        color_continuous_scale='RdBu',
        text=growth_data['Growth (%)'].apply(lambda x: f"{x:.1f}%")
    )
    
    fig.update_layout(
        title=f"Growth Analysis by {primary_dimension}",
        xaxis_title=primary_dimension,
        yaxis_title="Growth (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display growth data
    st.dataframe(
        growth_data,
        use_container_width=True,
        hide_index=True
    )

# Settings page
def settings_page():
    st.markdown("<h1 class='main-header'>Settings</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Data Management</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        st.info(f"Current dataset: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
        
        if st.button("Clear Current Data"):
            st.session_state.data = None
            st.success("Data cleared successfully!")
            st.rerun()
    
    st.markdown("<h2 class='sub-header'>Sample Data Generation</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate New Sample Data", use_container_width=True):
            with st.spinner("Generating sample data..."):
                st.session_state.data = generate_sample_data()
                st.success("Sample data generated successfully!")
                st.rerun()
    
    with col2:
        if st.session_state.data is not None:
            csv = st.session_state.data.to_csv(index=False)
            st.download_button(
                label="Download Current Data",
                data=csv,
                file_name=f"business_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    st.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This Business Intelligence Dashboard is a comprehensive tool for analyzing and visualizing business data. It provides:
    
    - **Dashboard**: Overview of key metrics and trends
    - **Sales Analysis**: Detailed analysis of sales performance
    - **Forecasting**: Predictive analytics for future performance
    - **Segment Analysis**: In-depth analysis of business segments
    
    The dashboard is built with Streamlit and uses various data science libraries for analysis and visualization.
    """)
    
    st.markdown("<h2 class='sub-header'>Help</h2>", unsafe_allow_html=True)
    
    with st.expander("How to use this dashboard"):
        st.markdown("""
        1. **Upload Data**: Use the sidebar to upload your CSV data or generate sample data
        2. **Navigate**: Use the sidebar buttons to navigate between different sections
        3. **Filter**: Use the filters at the top of each page to customize your analysis
        4. **Interact**: Hover over charts for more information, click on legends to filter
        5. **Download**: Download charts and data for your reports
        """)
    
    with st.expander("Data Format"):
        st.markdown("""
        The dashboard expects data with the following columns:
        
        - **Date**: Date of the transaction (YYYY-MM-DD)
        - **Category**: Product category
        - **Region**: Sales region
        - **Sales**: Sales amount
        - **Profit**: Profit amount
        - **Units**: Number of units sold
        - **Customers**: Number of customers
        
        You can upload your own data or use the sample data generator.
        """)

# Main app
def main():
    # Set up sidebar
    sidebar()
    
    # Render the selected page
    if st.session_state.page == 'dashboard':
        dashboard_page()
    elif st.session_state.page == 'sales':
        sales_analysis_page()
    elif st.session_state.page == 'forecast':
        forecasting_page()
    elif st.session_state.page == 'segment':
        segment_analysis_page()
    elif st.session_state.page == 'settings':
        settings_page()

if __name__ == "__main__":
    main()