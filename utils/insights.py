import pandas as pd
from typing import Dict, List, Any
import numpy as np

def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates high-level KPI metrics from the dataset.
    
    Args:
        df: Pandas DataFrame (after load_data cleaning, target Churn is 0/1)
        
    Returns:
        Dict: High-level KPI metrics.
    """
    total_customers = len(df)
    
    # Check target representation (in raw loading, Churn is converted to 1/0)
    # If not mapped, map it temporarily for safety
    if df['Churn'].dtype == object:
        churned_series = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0)
    else:
        churned_series = df['Churn']
        
    churned_count = int(churned_series.sum())
    active_count = total_customers - churned_count
    churn_rate = (churned_count / total_customers) * 100 if total_customers > 0 else 0.0
    
    avg_monthly_charges = float(df['MonthlyCharges'].mean())
    avg_tenure = float(df['tenure'].mean())
    
    # Estimate total revenue (sum of TotalCharges for all active and churned customers)
    total_revenue = float(df['TotalCharges'].sum())
    
    return {
        "total_customers": total_customers,
        "active_customers": active_count,
        "churned_customers": churned_count,
        "churn_rate": churn_rate,
        "avg_monthly_charges": avg_monthly_charges,
        "avg_tenure": avg_tenure,
        "total_revenue": total_revenue
    }

def get_segment_insights(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyzes the dataset to find high-risk customer segments and churn patterns.
    
    Returns:
        List of Dicts: Segment name, risk score, findings, and strategic recommendations.
    """
    insights = []
    
    # 1. Contract Type Analysis
    if 'Contract' in df.columns:
        contract_churn = df.groupby('Contract')['Churn'].mean() * 100
        m2m_churn = contract_churn.get('Month-to-month', 0.0)
        two_yr_churn = contract_churn.get('Two year', 0.0)
        
        if m2m_churn > 30: # typically month-to-month churn is very high (around 42%)
            insights.append({
                "segment": "Contract Type (Month-to-month)",
                "churn_rate": m2m_churn,
                "overall_avg": df['Churn'].mean() * 100,
                "finding": f"Month-to-Month contract customers have an extremely high churn rate of {m2m_churn:.1f}%, compared to just {two_yr_churn:.1f}% for customers on 2-year contracts.",
                "recommendation": "Implement an automated incentive campaign (e.g., '1 month free' or a 15% discount) to migrate Month-to-Month customers to 1-Year or 2-Year contracts. Provide a seamless one-click upgrade path in the customer portal.",
                "severity": "HIGH"
            })
            
    # 2. Internet Service Analysis
    if 'InternetService' in df.columns:
        internet_churn = df.groupby('InternetService')['Churn'].mean() * 100
        fiber_churn = internet_churn.get('Fiber optic', 0.0)
        dsl_churn = internet_churn.get('DSL', 0.0)
        
        if fiber_churn > dsl_churn: # Fiber optic usually has higher churn despite faster speeds (around 41%)
            insights.append({
                "segment": "Internet Service (Fiber Optic)",
                "churn_rate": fiber_churn,
                "overall_avg": df['Churn'].mean() * 100,
                "finding": f"Fiber Optic customers churn at a rate of {fiber_churn:.1f}%, significantly higher than DSL customers ({dsl_churn:.1f}%).",
                "recommendation": "This indicates potential issues with price sensitivity or service reliability. Launch a customer satisfaction survey specifically for Fiber Optic users, audit connection reliability in high-churn ZIP codes, and review competitive local pricing.",
                "severity": "HIGH"
            })

    # 3. Tenure Analysis
    if 'tenure' in df.columns:
        # Segment by tenure: Early (0-12 months), Mid (12-48 months), Long (48+ months)
        df_temp = df.copy()
        df_temp['TenureGroup'] = pd.cut(df_temp['tenure'], bins=[-1, 12, 24, 48, 72], labels=['0-12 Months', '12-24 Months', '24-48 Months', '48-72 Months'])
        tenure_churn = df_temp.groupby('TenureGroup', observed=False)['Churn'].mean() * 100
        early_churn = tenure_churn.get('0-12 Months', 0.0)
        
        if early_churn > 35:
            insights.append({
                "segment": "Tenure (First Year Customers)",
                "churn_rate": early_churn,
                "overall_avg": df['Churn'].mean() * 100,
                "finding": f"New customers in their first 12 months show a high churn rate of {early_churn:.1f}%. Risk drops dramatically after the first year.",
                "recommendation": "Strengthen the onboarding process. Launch proactive check-ins at months 1, 3, and 6. Provide tutorials for set-up, offer loyalty rewards at the 6-month mark, and ensure prompt technical support for early-stage support tickets.",
                "severity": "CRITICAL"
            })
            
    # 4. Tech Support Analysis
    if 'TechSupport' in df.columns:
        support_churn = df.groupby('TechSupport')['Churn'].mean() * 100
        no_support_churn = support_churn.get('No', 0.0)
        yes_support_churn = support_churn.get('Yes', 0.0)
        
        if no_support_churn > yes_support_churn * 1.5:
            insights.append({
                "segment": "Support Services (No Tech Support)",
                "churn_rate": no_support_churn,
                "overall_avg": df['Churn'].mean() * 100,
                "finding": f"Customers who do not have Tech Support churn at a rate of {no_support_churn:.1f}%, which is more than double the churn rate of customers with Tech Support ({yes_support_churn:.1f}%).",
                "recommendation": "Promote Tech Support as a high-value add-on or bundle it for free for the first 3-6 months. Send targeted emails highlighting the availability and ease of scheduling technical assistance.",
                "severity": "MEDIUM"
            })

    # 5. Payment Method Analysis
    if 'PaymentMethod' in df.columns:
        pay_churn = df.groupby('PaymentMethod')['Churn'].mean() * 100
        echeck_churn = pay_churn.get('Electronic check', 0.0)
        auto_churn = df[df['PaymentMethod'].str.contains('automatic', case=False)]['Churn'].mean() * 100
        
        if echeck_churn > auto_churn * 1.5:
            insights.append({
                "segment": "Billing Method (Electronic Check)",
                "churn_rate": echeck_churn,
                "overall_avg": df['Churn'].mean() * 100,
                "finding": f"Customers paying via Electronic Check churn at a rate of {echeck_churn:.1f}%, compared to just {auto_churn:.1f}% for automatic payment methods.",
                "recommendation": "Incentivize enrollment in Auto-Pay (Bank Transfer or Credit Card) by offering a one-time bill credit (e.g., $5 or $10) or a minor monthly discount. Auto-pay reduces involuntary churn due to expired cards or manually missed payments.",
                "severity": "MEDIUM"
            })

    return insights

def get_churn_risk_level(prob: float) -> Tuple[str, str, str]:
    """Translates a prediction probability into risk level, color code, and recommendation.
    
    Args:
        prob: Probability of churn (0.0 to 1.0)
        
    Returns:
        Tuple: (Risk Label, Color Hex, Executive Recommendation)
    """
    if prob < 0.2:
        return (
            "Low Risk", 
            "#10B981", 
            "Customer shows high loyalty indicators. Maintain standard service relationship and include in standard marketing communications."
        )
    elif prob < 0.5:
        return (
            "Moderate Risk", 
            "#F59E0B", 
            "Customer displays minor churn signals. Recommended action: Send a courtesy feedback survey or showcase new value-add products (e.g. streaming options, backup services) that fit their usage patterns."
        )
    elif prob < 0.8:
        return (
            "High Risk", 
            "#EF4444", 
            "Immediate proactive outreach recommended. Offer contract upgrade incentives (e.g. migrate to 1-year contract at discount) and check if they have active technical complaints."
        )
    else:
        return (
            "Critical Churn Threat", 
            "#991B1B", 
            "High risk of immediate churn. Recommended action: Direct account representative contact. Offer a significant subscription discount (e.g., 20% off for 6 months) or upgrade to premium support tiers at no cost to retain them."
        )
