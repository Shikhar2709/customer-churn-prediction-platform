import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

from utils.config import COLORS, PLOTLY_TEMPLATE

def plot_churn_distribution(df: pd.DataFrame) -> go.Figure:
    """Creates a donut chart of the Churn vs Non-Churn distribution.
    
    Args:
        df: Input DataFrame with Churn column.
        
    Returns:
        go.Figure: Donut chart figure.
    """
    # Count occurrences
    churn_counts = df['Churn'].value_counts()
    labels = ['Active Customers', 'Churned Customers']
    # If the target is already mapped to 1/0, make sure we align the indices
    values = [churn_counts.get(0, 0) + churn_counts.get('No', 0), 
              churn_counts.get(1, 0) + churn_counts.get('Yes', 0)]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.5,
        marker=dict(colors=[COLORS['active'], COLORS['churn']]),
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        textfont_size=14,
        pull=[0, 0.05]
    )])
    
    fig.update_layout(
        title_text="Overall Customer Churn Ratio",
        template=PLOTLY_TEMPLATE,
        height=350,
        margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig

def plot_numerical_distribution(df: pd.DataFrame, col: str, color_by_churn: bool = True) -> go.Figure:
    """Creates a histogram distribution for a numerical column.
    
    Args:
        df: Input DataFrame.
        col: The numerical column (e.g. tenure, MonthlyCharges).
        color_by_churn: Color bars differently for Churned vs Active.
        
    Returns:
        go.Figure: Histogram figure.
    """
    df_temp = df.copy()
    if color_by_churn:
        df_temp['Status'] = df_temp['Churn'].map({1: 'Churned', 0: 'Active'})
        fig = px.histogram(
            df_temp, 
            x=col, 
            color='Status',
            marginal="box",
            barmode="overlay",
            color_discrete_map={'Active': COLORS['active'], 'Churned': COLORS['churn']},
            opacity=0.75
        )
    else:
        fig = px.histogram(
            df_temp, 
            x=col, 
            marginal="box",
            color_discrete_sequence=[COLORS['primary']],
            opacity=0.8
        )
        
    fig.update_layout(
        title_text=f"Distribution of {col}",
        template=PLOTLY_TEMPLATE,
        height=400,
        margin=dict(t=50, b=50, l=50, r=20),
        xaxis_title=col,
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def plot_categorical_churn(df: pd.DataFrame, col: str) -> go.Figure:
    """Creates a grouped bar chart showing the churn rate across categorical levels.
    
    Args:
        df: Input DataFrame.
        col: Categorical column (e.g. Contract, InternetService).
        
    Returns:
        go.Figure: Grouped bar chart figure.
    """
    df_temp = df.copy()
    df_temp['Status'] = df_temp['Churn'].map({1: 'Churned', 0: 'Active'})
    
    # Calculate percentages for normalized viewing
    grouped = df_temp.groupby([col, 'Status'], observed=False).size().reset_index(name='Count')
    total = df_temp.groupby(col, observed=False).size().reset_index(name='Total')
    grouped = grouped.merge(total, on=col)
    grouped['Percentage'] = (grouped['Count'] / grouped['Total']) * 100
    
    fig = px.bar(
        grouped, 
        x=col, 
        y='Percentage', 
        color='Status',
        barmode='group',
        color_discrete_map={'Active': COLORS['active'], 'Churned': COLORS['churn']},
        hover_data=['Count', 'Total'],
        text=grouped['Percentage'].apply(lambda x: f"{x:.1f}%")
    )
    
    fig.update_traces(textposition='outside')
    
    fig.update_layout(
        title_text=f"Churn Distribution by {col}",
        template=PLOTLY_TEMPLATE,
        height=400,
        margin=dict(t=50, b=50, l=50, r=20),
        xaxis_title=col,
        yaxis_title="Percentage of Segment (%)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Generates a correlation heatmap of numerical features.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        go.Figure: Correlation heatmap figure.
    """
    # Select numeric features
    numeric_df = df[list(set(['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']) & set(df.columns))]
    corr = numeric_df.corr()
    
    # Format coefficients
    z = corr.values
    x = list(corr.columns)
    y = list(corr.index)
    
    fig = ff_create_annotated_heatmap_fallback(z, x, y)
    fig.update_layout(
        title_text="Numerical Feature Correlation Matrix",
        template=PLOTLY_TEMPLATE,
        height=380,
        margin=dict(t=60, b=20, l=80, r=20)
    )
    return fig

def ff_create_annotated_heatmap_fallback(z: np.ndarray, x: list, y: list) -> go.Figure:
    """Plots an annotated heatmap without importing plotly.figure_factory (which can be heavy)."""
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Tealrose',
        zmin=-1,
        zmax=1,
        hoverongaps=False
    ))
    
    # Add text annotations manually
    for i in range(len(y)):
        for j in range(len(x)):
            fig.add_annotation(
                x=x[j],
                y=y[i],
                text=f"{z[i][j]:.2f}",
                showarrow=False,
                font=dict(color="white" if abs(z[i][j]) > 0.4 else "black")
            )
            
    return fig

def plot_feature_importance(feat_imp_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Plots horizontal bar chart of feature importances.
    
    Args:
        feat_imp_df: DataFrame with 'Feature' and 'Importance' columns.
        top_n: Number of top features to display.
        
    Returns:
        go.Figure: Plotly horizontal bar chart.
    """
    top_imp = feat_imp_df.head(top_n).copy()
    top_imp = top_imp.sort_values(by='Importance', ascending=True) # Sort ascending for correct top-down drawing
    
    # Apply a gradient color scheme
    fig = px.bar(
        top_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Teal'
    )
    
    fig.update_layout(
        title_text=f"Top {top_n} Features Driving Customer Churn",
        template=PLOTLY_TEMPLATE,
        height=450,
        margin=dict(t=50, b=50, l=150, r=20),
        xaxis_title="Relative Influence Score",
        yaxis_title="Feature Name",
        coloraxis_showscale=False
    )
    return fig

def plot_confusion_matrix(conf_matrix: list) -> go.Figure:
    """Plots a confusion matrix heatmap.
    
    Args:
        conf_matrix: 2x2 nested list [[tn, fp], [fn, tp]].
        
    Returns:
        go.Figure: Plotly confusion matrix heatmap.
    """
    z = np.array(conf_matrix)
    x = ['Predicted Active', 'Predicted Churned']
    y = ['Actual Active', 'Actual Churned']
    
    # Total calculations for percentage display
    total = z.sum()
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Blues',
        showscale=False
    ))
    
    # Annotate with values and percentage of total
    for i in range(2):
        for j in range(2):
            val = z[i][j]
            pct = (val / total) * 100
            # Flip i coordinate for layout match
            fig.add_annotation(
                x=x[j],
                y=y[i],
                text=f"<b>{val}</b><br>({pct:.1f}%)",
                showarrow=False,
                font=dict(color="white" if val > z.max()/2 else "black", size=15)
            )
            
    fig.update_layout(
        title_text="Confusion Matrix",
        template=PLOTLY_TEMPLATE,
        height=350,
        margin=dict(t=50, b=40, l=80, r=20),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed")
    )
    return fig

def plot_roc_curve(y_test: List[int], y_prob: List[float]) -> go.Figure:
    """Generates the ROC curve visualization.
    
    Args:
        y_test: Actual labels.
        y_prob: Predicted churn probabilities.
        
    Returns:
        go.Figure: Plotly line chart.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    fig = go.Figure()
    
    # Standard diagonal baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Random Classifier (AUC = 0.50)'
    ))
    
    # ROC Line
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        line=dict(color=COLORS['primary'], width=3),
        name=f'Model ROC Curve (AUC = {auc:.3f})'
    ))
    
    fig.update_layout(
        title_text=f"ROC Curve (AUC: {auc:.3f})",
        template=PLOTLY_TEMPLATE,
        height=400,
        margin=dict(t=50, b=50, l=50, r=20),
        xaxis=dict(title='False Positive Rate (1 - Specificity)', range=[0, 1.05]),
        yaxis=dict(title='True Positive Rate (Sensitivity)', range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def plot_pr_curve(y_test: List[int], y_prob: List[float]) -> go.Figure:
    """Generates the Precision-Recall curve visualization.
    
    Args:
        y_test: Actual labels.
        y_prob: Predicted churn probabilities.
        
    Returns:
        go.Figure: Plotly line chart.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    fig = go.Figure()
    
    # PR Line
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        line=dict(color=COLORS['secondary'], width=3),
        name='Precision-Recall Curve'
    ))
    
    fig.update_layout(
        title_text="Precision-Recall Curve",
        template=PLOTLY_TEMPLATE,
        height=400,
        margin=dict(t=50, b=50, l=50, r=20),
        xaxis=dict(title='Recall (Sensitivity)', range=[0, 1.05]),
        yaxis=dict(title='Precision (Positive Predictive Value)', range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig
