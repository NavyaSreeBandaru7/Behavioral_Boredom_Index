import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List
import logging

# Configure page
st.set_page_config(
    page_title="Behavioral Boredom Index Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .high-risk {
        border-left-color: #ff4757;
    }
    .medium-risk {
        border-left-color: #ffa502;
    }
    .low-risk {
        border-left-color: #2ed573;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 12px;
        padding-right: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

if 'auth_token' not in st.session_state:
    st.session_state.auth_token = "bbi-demo-token-2024"

if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = []

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# Utility functions
def generate_sample_data():
    """Generate realistic sample data for demo"""
    np.random.seed(42)
    
    employees = [f"EMP{i:03d}" for i in range(1, 51)]
    departments = ["Engineering", "Sales", "Marketing", "HR", "Operations"]
    
    data = []
    for emp in employees:
        dept = np.random.choice(departments)
        base_score = {
            "Engineering": 0.45,
            "Sales": 0.35,
            "Marketing": 0.40,
            "HR": 0.30,
            "Operations": 0.50
        }[dept]
        
        boredom_score = max(0, min(1, base_score + np.random.normal(0, 0.15)))
        
        data.append({
            "employee_id": emp,
            "department": dept,
            "boredom_score": boredom_score,
            "risk_level": "LOW" if boredom_score < 0.3 else "MEDIUM" if boredom_score < 0.6 else "HIGH" if boredom_score < 0.8 else "CRITICAL",
            "last_analysis": datetime.now() - timedelta(hours=np.random.randint(1, 48)),
            "confidence": np.random.uniform(0.7, 0.95),
            "turnover_risk": min(1, boredom_score * 1.2 + np.random.normal(0, 0.1))
        })
    
    return pd.DataFrame(data)

def create_risk_distribution_chart(df):
    """Create risk distribution donut chart"""
    risk_counts = df['risk_level'].value_counts()
    
    colors = {
        'LOW': '#2ed573',
        'MEDIUM': '#ffa502', 
        'HIGH': '#ff6b6b',
        'CRITICAL': '#ff4757'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.5,
        marker_colors=[colors.get(label, '#gray') for label in risk_counts.index],
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Employee Risk Distribution",
        font_size=14,
        showlegend=True,
        height=400
    )
    
    return fig

def create_department_comparison_chart(df):
    """Create department comparison chart"""
    dept_stats = df.groupby('department').agg({
        'boredom_score': 'mean',
        'turnover_risk': 'mean',
        'employee_id': 'count'
    }).round(3)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Boredom Score', 'Average Turnover Risk'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Boredom scores
    fig.add_trace(
        go.Bar(
            x=dept_stats.index,
            y=dept_stats['boredom_score'],
            name="Boredom Score",
            marker_color='#ff6b6b',
            text=dept_stats['boredom_score'],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Turnover risk
    fig.add_trace(
        go.Bar(
            x=dept_stats.index,
            y=dept_stats['turnover_risk'],
            name="Turnover Risk",
            marker_color='#ffa502',
            text=dept_stats['turnover_risk'],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Department Comparison Analysis"
    )
    
    return fig

def create_trend_chart():
    """Create engagement trend chart"""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    # Simulate trend data
    base_trend = 0.4
    trend_data = []
    
    for i, date in enumerate(dates):
        # Add weekly pattern and random noise
        weekly_pattern = 0.1 * np.sin(2 * np.pi * i / 7)
        noise = np.random.normal(0, 0.05)
        score = base_trend + weekly_pattern + noise + (i * 0.003)  # Slight upward trend
        
        trend_data.append({
            'date': date,
            'avg_boredom_score': max(0, min(1, score)),
            'employee_count': np.random.randint(45, 52)
        })
    
    trend_df = pd.DataFrame(trend_data)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Boredom score trend
    fig.add_trace(
        go.Scatter(
            x=trend_df['date'],
            y=trend_df['avg_boredom_score'],
            mode='lines+markers',
            name='Avg Boredom Score',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=6)
        ),
        secondary_y=False,
    )
    
    # Employee count
    fig.add_trace(
        go.Bar(
            x=trend_df['date'],
            y=trend_df['employee_count'],
            name='Employees Analyzed',
            opacity=0.3,
            marker_color='#70a1ff'
        ),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxis(title_text="Average Boredom Score", secondary_y=False)
    fig.update_yaxis(title_text="Employees Analyzed", secondary_y=True)
    
    fig.update_layout(
        title="30-Day Engagement Trend",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_correlation_heatmap(df):
    """Create feature correlation heatmap"""
    # Simulate additional features for correlation
    np.random.seed(42)
    correlation_data = pd.DataFrame({
        'Boredom Score': df['boredom_score'],
        'Turnover Risk': df['turnover_risk'],
        'Email Response Time': np.random.normal(0.5, 0.2, len(df)),
        'Calendar Density': np.random.normal(0.6, 0.15, len(df)),
        'Collaboration Score': 1 - df['boredom_score'] + np.random.normal(0, 0.1, len(df)),
        'Innovation Indicators': 1 - df['boredom_score'] * 0.8 + np.random.normal(0, 0.1, len(df))
    })
    
    corr_matrix = correlation_data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdYlBu_r',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Feature Correlation Analysis",
        height=500,
        width=600
    )
    
    return fig

def create_individual_employee_chart(employee_data):
    """Create individual employee analysis chart"""
    # Simulate historical data for the employee
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    base_score = employee_data['boredom_score']
    
    historical_scores = []
    for i in range(90):
        # Add realistic variance
        daily_variance = np.random.normal(0, 0.05)
        trend_factor = (i - 45) * 0.002  # Gradual change over time
        score = max(0, min(1, base_score + daily_variance + trend_factor))
        historical_scores.append(score)
    
    fig = go.Figure()
    
    # Historical trend
    fig.add_trace(go.Scatter(
        x=dates,
        y=historical_scores,
        mode='lines',
        name='Boredom Score',
        line=dict(color='#ff6b6b', width=2),
        fill='tonexty'
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.3, line_dash="dash", line_color="green", 
                  annotation_text="Low Risk Threshold")
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Risk Threshold")
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                  annotation_text="High Risk Threshold")
    
    fig.update_layout(
        title=f"90-Day Engagement Trend - {employee_data['employee_id']}",
        xaxis_title="Date",
        yaxis_title="Boredom Score",
        height=400,
        showlegend=True
    )
    
    return fig

# Main Dashboard
def main():
    # Header
    st.title("🎯 Behavioral Boredom Index Dashboard")
    st.markdown("**Advanced Employee Engagement Analytics with Privacy-Preserving AI**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Real-time toggle
        real_time_enabled = st.toggle("Real-time Monitoring", value=True)
        
        if real_time_enabled:
            refresh_interval = st.selectbox(
                "Refresh Interval",
                ["30 seconds", "1 minute", "5 minutes", "15 minutes"],
                index=1
            )
            
            # Auto-refresh logic
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        st.divider()
        
        # Filters
        st.subheader("📊 Filters")
        
        departments = ["All", "Engineering", "Sales", "Marketing", "HR", "Operations"]
        selected_dept = st.selectbox("Department", departments)
        
        risk_levels = ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        selected_risk = st.selectbox("Risk Level", risk_levels)
        
        time_range = st.selectbox(
            "Time Range",
            ["Last 7 days", "Last 30 days", "Last 90 days", "Last 6 months"]
        )
        
        st.divider()
        
        # Alert settings
        st.subheader("🚨 Alert Thresholds")
        
        high_risk_threshold = st.slider(
            "High Risk Alert (%)",
            min_value=10,
            max_value=50,
            value=25,
            help="Trigger alert when % of team is high risk"
        )
        
        turnover_risk_threshold = st.slider(
            "Turnover Risk Alert (%)",
            min_value=5,
            max_value=30,
            value=15,
            help="Trigger alert for predicted turnover risk"
        )
    
    # Generate sample data
    df = generate_sample_data()
    
    # Apply filters
    if selected_dept != "All":
        df = df[df['department'] == selected_dept]
    
    if selected_risk != "All":
        df = df[df['risk_level'] == selected_risk]
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "👥 Team Analysis", 
        "👤 Individual Analysis", 
        "📈 Trends & Forecasting", 
        "⚙️ Model Management"
    ])
    
    with tab1:
        # Overview tab
        st.header("Executive Overview")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_employees = len(df)
            st.metric(
                label="Total Employees",
                value=total_employees,
                delta=f"+{np.random.randint(1, 5)} this week"
            )
        
        with col2:
            avg_boredom = df['boredom_score'].mean()
            st.metric(
                label="Avg Boredom Score",
                value=f"{avg_boredom:.2f}",
                delta=f"{np.random.uniform(-0.05, 0.05):.3f}"
            )
        
        with col3:
            high_risk_count = len(df[df['risk_level'].isin(['HIGH', 'CRITICAL'])])
            high_risk_pct = (high_risk_count / total_employees) * 100
            st.metric(
                label="High Risk Employees",
                value=f"{high_risk_count} ({high_risk_pct:.1f}%)",
                delta=f"{np.random.randint(-2, 3)} from last week"
            )
        
        with col4:
            avg_turnover_risk = df['turnover_risk'].mean()
            st.metric(
                label="Avg Turnover Risk",
                value=f"{avg_turnover_risk:.1%}",
                delta=f"{np.random.uniform(-0.02, 0.02):.1%}"
            )
        
        with col5:
            prediction_accuracy = 0.87  # Simulated
            st.metric(
                label="Model Accuracy",
                value=f"{prediction_accuracy:.1%}",
                delta="+1.2% this month"
            )
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_risk_distribution_chart(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_department_comparison_chart(df), use_container_width=True)
        
        # Trend chart
        st.plotly_chart(create_trend_chart(), use_container_width=True)
        
        # Alerts section
        st.header("🚨 Active Alerts")
        
        # Check thresholds
        current_high_risk_pct = (high_risk_count / total_employees) * 100
        
        if current_high_risk_pct > high_risk_threshold:
            st.error(f"⚠️ HIGH RISK ALERT: {high_risk_pct:.1f}% of employees are high risk (threshold: {high_risk_threshold}%)")
        
        if avg_turnover_risk > (turnover_risk_threshold / 100):
            st.warning(f"⚠️ TURNOVER ALERT: Average turnover risk is {avg_turnover_risk:.1%} (threshold: {turnover_risk_threshold}%)")
        
        # Recent activity
        st.header("📋 Recent Activity")
        
        recent_activity = [
            {"time": "2 hours ago", "event": "High-risk employee identified in Engineering", "severity": "high"},
            {"time": "4 hours ago", "event": "Team engagement score improved in Sales", "severity": "low"},
            {"time": "6 hours ago", "event": "Model retrained with new data", "severity": "medium"},
            {"time": "1 day ago", "event": "Weekly analysis completed for all departments", "severity": "low"}
        ]
        
        for activity in recent_activity:
            severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[activity["severity"]]
            st.write(f"{severity_color} **{activity['time']}** - {activity['event']}")
    
    with tab2:
        # Team Analysis tab
        st.header("👥 Team Analysis")
        
        # Department selector for detailed analysis
        selected_dept_detail = st.selectbox(
            "Select Department for Detailed Analysis",
            df['department'].unique(),
            key="dept_detail"
        )
        
        dept_df = df[df['department'] == selected_dept_detail]
        
        # Department metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Team Size",
                len(dept_df),
                help="Number of employees in department"
            )
        
        with col2:
            dept_avg_boredom = dept_df['boredom_score'].mean()
            overall_avg = df['boredom_score'].mean()
            delta_boredom = dept_avg_boredom - overall_avg
            st.metric(
                "Avg Boredom Score",
                f"{dept_avg_boredom:.2f}",
                delta=f"{delta_boredom:+.3f} vs company avg"
            )
        
        with col3:
            dept_high_risk = len(dept_df[dept_df['risk_level'].isin(['HIGH', 'CRITICAL'])])
            st.metric(
                "High Risk Count",
                dept_high_risk,
                delta=f"{(dept_high_risk/len(dept_df)*100):.1f}% of team"
            )
        
        with col4:
            dept_turnover_risk = dept_df['turnover_risk'].mean()
            st.metric(
                "Turnover Risk",
                f"{dept_turnover_risk:.1%}",
                help="Predicted probability of turnover"
            )
        
        # Team heatmap
        st.subheader("Team Risk Heatmap")
        
        # Create a grid layout for team visualization
        team_size = len(dept_df)
        cols_per_row = 8
        rows = (team_size + cols_per_row - 1) // cols_per_row
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                emp_idx = row * cols_per_row + col_idx
                if emp_idx < team_size:
                    emp = dept_df.iloc[emp_idx]
                    risk_color = {
                        'LOW': '🟢',
                        'MEDIUM': '🟡', 
                        'HIGH': '🟠',
                        'CRITICAL': '🔴'
                    }[emp['risk_level']]
                    
                    with cols[col_idx]:
                        st.write(f"{risk_color}")
                        st.caption(f"{emp['employee_id']}")
        
        # Detailed employee table
        st.subheader("Detailed Employee Analysis")
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            ["Boredom Score (High to Low)", "Turnover Risk (High to Low)", "Employee ID", "Risk Level"]
        )
        
        if sort_by == "Boredom Score (High to Low)":
            display_df = dept_df.sort_values('boredom_score', ascending=False)
        elif sort_by == "Turnover Risk (High to Low)":
            display_df = dept_df.sort_values('turnover_risk', ascending=False)
        elif sort_by == "Employee ID":
            display_df = dept_df.sort_values('employee_id')
        else:
            display_df = dept_df.sort_values('risk_level')
        
        # Style the dataframe
        def style_risk_level(val):
            colors = {
                'LOW': 'background-color: #d4edda',
                'MEDIUM': 'background-color: #fff3cd',
                'HIGH': 'background-color: #f8d7da',
                'CRITICAL': 'background-color: #f5c6cb'
            }
            return colors.get(val, '')
        
        styled_df = display_df[['employee_id', 'boredom_score', 'risk_level', 'turnover_risk', 'confidence']].style.applymap(
            style_risk_level, subset=['risk_level']
        ).format({
            'boredom_score': '{:.3f}',
            'turnover_risk': '{:.1%}',
            'confidence': '{:.2f}'
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Team recommendations
        st.subheader("🎯 Team Recommendations")
        
        high_risk_employees = len(dept_df[dept_df['risk_level'].isin(['HIGH', 'CRITICAL'])])
        avg_score = dept_df['boredom_score'].mean()
        
        recommendations = []
        
        if high_risk_employees > len(dept_df) * 0.3:
            recommendations.append("🚨 **Immediate Action Required**: Over 30% of team is high risk")
            recommendations.append("📅 Schedule emergency team meeting to address engagement issues")
        
        if avg_score > 0.6:
            recommendations.append("💡 Consider team-wide engagement initiatives")
            recommendations.append("🔄 Review current project assignments and workload distribution")
        
        if dept_df['turnover_risk'].max() > 0.8:
            recommendations.append("🎯 Conduct one-on-one meetings with highest risk employees")
            recommendations.append("💰 Review compensation and career development opportunities")
        
        if not recommendations:
            recommendations.append("✅ Team engagement levels are healthy")
            recommendations.append("📈 Continue current management practices")
        
        for rec in recommendations:
            st.write(rec)
    
    with tab3:
        # Individual Analysis tab
        st.header("👤 Individual Employee Analysis")
        
        # Employee selector
        selected_employee = st.selectbox(
            "Select Employee",
            df['employee_id'].tolist(),
            key="individual_employee"
        )
        
        employee_data = df[df['employee_id'] == selected_employee].iloc[0]
        
        # Employee overview
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Employee Profile")
            
            risk_emoji = {
                'LOW': '🟢',
                'MEDIUM': '🟡',
                'HIGH': '🟠', 
                'CRITICAL': '🔴'
            }[employee_data['risk_level']]
            
            st.write(f"**Employee ID:** {employee_data['employee_id']}")
            st.write(f"**Department:** {employee_data['department']}")
            st.write(f"**Risk Level:** {risk_emoji} {employee_data['risk_level']}")
            st.write(f"**Last Analysis:** {employee_data['last_analysis'].strftime('%Y-%m-%d %H:%M')}")
            
            # Key metrics
            st.metric("Boredom Score", f"{employee_data['boredom_score']:.3f}")
            st.metric("Turnover Risk", f"{employee_data['turnover_risk']:.1%}")
            st.metric("Confidence", f"{employee_data['confidence']:.2f}")
        
        with col2:
            # Individual trend chart
            st.plotly_chart(create_individual_employee_chart(employee_data), use_container_width=True)
        
        # Detailed analysis
        st.subheader("🔍 Detailed Analysis")
        
        # Simulate detailed factors
        factors = {
            "Email Response Latency": np.random.uniform(0.2, 0.8),
            "Calendar Emptiness": np.random.uniform(0.1, 0.7),
            "Collaboration Level": np.random.uniform(0.3, 0.9),
            "Innovation Indicators": np.random.uniform(0.2, 0.8),
            "Sentiment Score": np.random.uniform(0.3, 0.8),
            "Focus Patterns": np.random.uniform(0.4, 0.9)
        }
        
        # Create radar chart for factors
        categories = list(factors.keys())
        values = list(factors.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=employee_data['employee_id'],
            line_color='#ff6b6b'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Behavioral Factors Analysis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("📋 Personalized Recommendations")
        
        if employee_data['risk_level'] == 'CRITICAL':
            recs = [
                "🚨 **Immediate intervention required**",
                "📅 Schedule urgent one-on-one meeting within 24 hours",
                "🔄 Consider role adjustment or project reassignment",
                "💡 Explore professional development opportunities",
                "👥 Assign mentor or buddy for support"
            ]
        elif employee_data['risk_level'] == 'HIGH':
            recs = [
                "⚠️ **High priority attention needed**",
                "📅 Schedule check-in meeting within 48 hours",
                "🎯 Review current workload and responsibilities",
                "📈 Discuss career goals and growth opportunities",
                "🤝 Increase collaboration on interesting projects"
            ]
        elif employee_data['risk_level'] == 'MEDIUM':
            recs = [
                "📊 **Monitor engagement trends**",
                "💡 Provide new challenges or learning opportunities",
                "🎮 Consider temporary project rotation",
                "📝 Schedule regular feedback sessions",
                "🏆 Recognize current contributions"
            ]
        else:
            recs = [
                "✅ **Engagement levels are healthy**",
                "🌟 Consider for leadership opportunities",
                "👨‍🏫 Use as mentor for struggling team members",
                "🏆 Recognize and reward performance",
                "📈 Maintain current trajectory"
            ]
        
        for rec in recs:
            st.write(rec)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📧 Send Engagement Survey"):
                st.success("Engagement survey sent!")
        
        with col2:
            if st.button("📅 Schedule Meeting"):
                st.success("Meeting scheduled!")
        
        with col3:
            if st.button("📊 Generate Report"):
                st.success("Report generated!")
    
    with tab4:
        # Trends & Forecasting tab
        st.header("📈 Trends & Forecasting")
        
        # Time series analysis
        st.subheader("Historical Trends")
        
        # Generate historical data
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
        
        # Simulate organizational trends
        base_trend = 0.4
        seasonal_pattern = 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        random_noise = np.random.normal(0, 0.02, len(dates))
        long_term_trend = np.linspace(0, 0.05, len(dates))  # Slight upward trend
        
        boredom_trend = base_trend + seasonal_pattern + random_noise + long_term_trend
        boredom_trend = np.clip(boredom_trend, 0, 1)
        
        trend_df = pd.DataFrame({
            'date': dates,
            'boredom_score': boredom_trend,
            'employees_analyzed': np.random.randint(45, 55, len(dates))
        })
        
        # Create comprehensive trend chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Boredom Score Trend', 'Employee Coverage'),
            vertical_spacing=0.1
        )
        
        # Boredom trend
        fig.add_trace(
            go.Scatter(
                x=trend_df['date'],
                y=trend_df['boredom_score'],
                mode='lines',
                name='Daily Average',
                line=dict(color='#ff6b6b', width=1)
            ),
            row=1, col=1
        )
        
        # 7-day moving average
        trend_df['ma_7'] = trend_df['boredom_score'].rolling(window=7).mean()
        fig.add_trace(
            go.Scatter(
                x=trend_df['date'],
                y=trend_df['ma_7'],
                mode='lines',
                name='7-Day Average',
                line=dict(color='#ff4757', width=3)
            ),
            row=1, col=1
        )
        
        # Employee coverage
        fig.add_trace(
            go.Bar(
                x=trend_df['date'],
                y=trend_df['employees_analyzed'],
                name='Employees Analyzed',
                marker_color='#70a1ff',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Boredom Score", row=1, col=1)
        fig.update_yaxes(title_text="Employee Count", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecasting
        st.subheader("🔮 Predictive Forecasting")
        
        forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
        
        # Simple forecasting simulation
        last_value = trend_df['boredom_score'].iloc[-1]
        forecast_dates = pd.date_range(start=trend_df['date'].iloc[-1] + timedelta(days=1), 
                                     periods=forecast_days, freq='D')
        
        # Linear trend + seasonal component
        trend_slope = (trend_df['boredom_score'].iloc[-30:].mean() - 
                      trend_df['boredom_score'].iloc[-60:-30].mean()) / 30
        
        forecast_values = []
        for i in range(forecast_days):
            seasonal = 0.05 * np.sin(2 * np.pi * i / 365.25)
            trend = trend_slope * i
            uncertainty = np.random.normal(0, 0.01)
            forecast_val = last_value + trend + seasonal + uncertainty
            forecast_values.append(np.clip(forecast_val, 0, 1))
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast_values,
            'lower_bound': [f - 0.05 for f in forecast_values],
            'upper_bound': [f + 0.05 for f in forecast_values]
        })
        
        # Forecast chart
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=trend_df['date'].tail(60),
            y=trend_df['boredom_score'].tail(60),
            mode='lines',
            name='Historical',
            line=dict(color='#2f3542')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='#ff6b6b', dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=list(forecast_df['date']) + list(forecast_df['date'][::-1]),
            y=list(forecast_df['upper_bound']) + list(forecast_df['lower_bound'][::-1]),
            fill='toself',
            fillcolor='rgba(255, 107, 107, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"{forecast_days}-Day Boredom Score Forecast",
            xaxis_title="Date",
            yaxis_title="Boredom Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("🎯 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Trend Analysis:**")
            current_trend = "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
            st.write(f"• Current trend: {current_trend}")
            st.write(f"• Seasonal pattern: {'High variance' if np.std(seasonal_pattern) > 0.05 else 'Low variance'}")
            st.write(f"• Forecast confidence: {'High' if np.std(forecast_values) < 0.1 else 'Medium'}")
        
        with col2:
            st.write("**Risk Predictions:**")
            future_high_risk_days = sum(1 for f in forecast_values if f > 0.6)
            st.write(f"• High-risk days forecasted: {future_high_risk_days}/{forecast_days}")
            st.write(f"• Peak risk period: {forecast_dates[np.argmax(forecast_values)].strftime('%Y-%m-%d')}")
            st.write(f"• Intervention recommended: {'Yes' if max(forecast_values) > 0.7 else 'Monitor'}")
    
    with tab5:
        # Model Management tab
        st.header("⚙️ Model Management")
        
        # Model status
        st.subheader("🤖 Model Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Version", "v2.1.3")
            st.metric("Last Trained", "2024-08-15")
            st.metric("Training Data Size", "125,000 samples")
        
        with col2:
            st.metric("Accuracy", "87.3%")
            st.metric("Precision", "84.1%") 
            st.metric("Recall", "89.7%")
        
        with col3:
            st.metric("F1-Score", "86.8%")
            st.metric("AUC-ROC", "0.923")
            st.metric("Inference Time", "45ms")
        
        # Model performance
        st.subheader("📊 Model Performance")
        
        # Confusion matrix simulation
        confusion_data = np.array([[120, 15, 3], [8, 95, 12], [2, 7, 78]])
        confusion_labels = ['Low Risk', 'Medium Risk', 'High Risk']
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_data,
            x=confusion_labels,
            y=confusion_labels,
            colorscale='Blues',
            text=confusion_data,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("🎯 Feature Importance")
        
        features = [
            "Email Response Latency", "Calendar Density", "Collaboration Score",
            "Document Edit Frequency", "Sentiment Analysis", "Meeting Participation",
            "After-hours Activity", "Innovation Indicators"
        ]
        
        importance_scores = np.random.uniform(0.1, 0.9, len(features))
        importance_scores = importance_scores / importance_scores.sum()
        
        fig = go.Figure(go.Bar(
            x=importance_scores,
            y=features,
            orientation='h',
            marker_color='#70a1ff'
        ))
        
        fig.update_layout(
            title="Feature Importance Scores",
            xaxis_title="Importance Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model actions
        st.subheader("🔧 Model Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Retrain Model"):
                with st.spinner("Retraining model..."):
                    time.sleep(3)  # Simulate training time
                st.success("Model retrained successfully!")
        
        with col2:
            if st.button("📊 Run Validation"):
                with st.spinner("Running validation..."):
                    time.sleep(2)
                st.success("Validation completed! Accuracy: 87.8%")
        
        with col3:
            if st.button("🚀 Deploy Model"):
                st.success("Model deployed to production!")
        
        # Federated Learning Status
        st.subheader("🌐 Federated Learning Status")
        
        fl_metrics = {
            "Active Clients": 12,
            "Training Rounds": 156,
            "Data Privacy Score": "100%",
            "Convergence Status": "Converged",
            "Last Aggregation": "2 hours ago"
        }
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        for i, (metric, value) in enumerate(fl_metrics.items()):
            with [col1, col2, col3, col4, col5][i]:
                st.metric(metric, value)
        
        # Data quality monitoring
        st.subheader("📈 Data Quality Monitoring")
        
        quality_metrics = pd.DataFrame({
            'Date': pd.date_range(end=datetime.now(), periods=7, freq='D'),
            'Completeness': np.random.uniform(0.85, 0.98, 7),
            'Consistency': np.random.uniform(0.90, 0.99, 7),
            'Accuracy': np.random.uniform(0.88, 0.97, 7)
        })
        
        fig = go.Figure()
        
        for metric in ['Completeness', 'Consistency', 'Accuracy']:
            fig.add_trace(go.Scatter(
                x=quality_metrics['Date'],
                y=quality_metrics[metric],
                mode='lines+markers',
                name=metric,
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="Data Quality Trends (7 days)",
            xaxis_title="Date",
            yaxis_title="Quality Score",
            yaxis=dict(range=[0.8, 1.0]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.divider()
    st.markdown("*Behavioral Boredom Index Dashboard - Built with ❤️ using Streamlit, Plotly, and advanced ML*")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
