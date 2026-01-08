import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from churn_backend import ChurnPredictionBackend
import os
import tempfile
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🎯 Advanced Churn Prediction Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'backend' not in st.session_state:
    st.session_state.backend = ChurnPredictionBackend()
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'feature_types' not in st.session_state:
    st.session_state.feature_types = {}

backend = st.session_state.backend

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🎯 Churn Prediction System")
    st.markdown("**Advanced ML Platform**")
    st.divider()
    
    # Configuration Status
    st.subheader("⚙️ Configuration")
    gemini_key = os.getenv('GEMINI_API_KEY', '')
    if gemini_key and not gemini_key.startswith('YOUR_'):
        st.success("✅ Gemini API Key Loaded from .env")
    else:
        st.warning("⚠️ Gemini API Key not found. Add to .env file for LLM features")
    
    st.divider()
    
    # Quick Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset All", help="Clear all data and models"):
            st.session_state.trained_models = False
            st.session_state.current_prediction = None
            st.rerun()
    
    with col2:
        if st.button("ℹ️ About", help="Platform information"):
            st.info("""
            🚀 Features:
            - 4 ML Models (LR, RF, XGB, GB)
            - SHAP Explainability
            - Comprehensive Reports
            - Drift Detection
            - Feedback Loop
            """)
    
    st.divider()
    
    # Model Status
    if st.session_state.trained_models and backend.model_metrics:
        st.subheader("📊 Model Status")
        for model_name, metrics in backend.model_metrics.items():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.caption(f"**{model_name}**")
            with col2:
                st.metric("F1", f"{metrics['f1']:.2f}", label_visibility="collapsed")

# ==================== MAIN CONTENT ====================
st.title("🎯 Advanced Customer Churn Prediction")
st.markdown("*Predict, Explain, and Act on Customer Churn with AI-Powered Insights*")

# ==================== TRAINING PAGE ====================
if not st.session_state.trained_models or backend.training_data is None:
    st.header("📊 Step 1: Train Models")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Upload Training Data")
        uploaded_file = st.file_uploader(
            "Upload CSV (Telco Churn format)",
            type=['csv'],
            help="Dataset should contain 'churn' column"
        )
        
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(uploaded_file.getbuffer())
                csv_path = tmp.name
            
            df_preview = pd.read_csv(csv_path)
            st.caption(f"📈 Rows: {len(df_preview)} | Columns: {len(df_preview.columns)}")
            
            with st.expander("👀 Preview Data"):
                st.dataframe(df_preview.head(10), use_container_width=True)
            
            if st.button("🚀 Train All Models (LR, RF, XGB, GB)", use_container_width=True, type="primary"):
                with st.spinner("📊 Training 4 models..."):
                    result = backend.train_models(csv_path)
                    if result['success']:
                        st.session_state.trained_models = True
                        st.session_state.feature_types = backend.feature_types
                        st.balloons()
                        st.success("✅ All models trained successfully!")
                        for log_line in result['log']:
                            st.caption(log_line)
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")
    
    with col2:
        st.subheader("📋 Template Info")
        st.info("""
        **Required:**
        - `churn`: Yes/No or 1/0
        - Numeric: tenure, charges
        - Categorical: contract, service
        
        **Min:** 100 rows
        """)

# ==================== MAIN DASHBOARD ====================
else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Prediction",
        "💰 Business Impact",
        "🔍 Explainability",
        "📈 Model Comparison",
        "⚠️ Monitoring",
        "📄 Reports"
    ])
    
    # ==================== TAB 1: PREDICTION ====================
    with tab1:
        st.header("🎯 Customer Churn Prediction")
        
        col_select, col_info = st.columns([1, 2])
        with col_select:
            selected_model = st.selectbox(
                "🤖 Select Model:",
                list(backend.models.keys()),
                index=3  # Default to GradientBoosting
            )
        with col_info:
            best_model = backend._get_best_model()
            st.info(f"🏆 Best Model: **{best_model}** (by F1-Score)")
        
        st.divider()
        st.subheader("📝 Enter Customer Details")
        
        input_data = {}
        
        # Dynamic input generation based on feature types
        if backend.feature_columns:
            # Separate numeric and categorical columns
            numeric_cols = [c for c in backend.feature_columns if backend.feature_types.get(c) == 'numeric']
            categorical_cols = [c for c in backend.feature_columns if backend.feature_types.get(c) == 'categorical']
            
            # Numeric inputs
            if numeric_cols:
                st.subheader("📊 Numeric Features")
                cols = st.columns(3)
                for idx, col in enumerate(numeric_cols[:6]):
                    with cols[idx % 3]:
                        if 'tenure' in col.lower():
                            input_data[col] = st.slider(col, 0, 72, 24)
                        elif 'charge' in col.lower():
                            input_data[col] = st.slider(col, 0.0, 200.0, 65.0)
                        else:
                            input_data[col] = st.number_input(col, value=0.0)
                
                if len(numeric_cols) > 6:
                    with st.expander("➕ More Numeric Features"):
                        for col in numeric_cols[6:]:
                            input_data[col] = st.number_input(col, value=0.0, key=f"num_{col}")
            
            # Categorical inputs
            if categorical_cols:
                st.subheader("📋 Categorical Features")
                cols = st.columns(3)
                for idx, col in enumerate(categorical_cols[:3]):
                    with cols[idx % 3]:
                        unique_vals = backend.training_data[col].unique().tolist()
                        input_data[col] = st.selectbox(col, unique_vals)
                
                if len(categorical_cols) > 3:
                    with st.expander("➕ More Categorical Features"):
                        for col in categorical_cols[3:]:
                            unique_vals = backend.training_data[col].unique().tolist()
                            input_data[col] = st.selectbox(col, unique_vals, key=f"extra_{col}")
        
        st.divider()
        
        if st.button("🔮 Predict & Generate Report", use_container_width=True, type="primary"):
            with st.spinner("📊 Generating comprehensive analysis..."):
                result = backend.generate_comprehensive_report(input_data, selected_model)
                
                if result['success']:
                    st.session_state.current_prediction = result
                    
                    # Display prediction results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        churn_status = "🚨 HIGH RISK" if result['prediction']['churn_prediction'] == 1 else "✅ LOW RISK"
                        st.metric(
                            f"{selected_model}",
                            churn_status,
                            f"{result['prediction']['churn_probability']:.1%}"
                        )
                    
                    with col2:
                        best_status = "🚨 HIGH RISK" if result['prediction']['best_model_prediction'] == 1 else "✅ LOW RISK"
                        st.metric(
                            f"{result['prediction']['best_model_name']}",
                            best_status,
                            f"{result['prediction']['best_model_probability']:.1%}"
                        )
                    
                    with col3:
                        agreement_text = "✅ AGREE" if result['prediction']['agreement'] else "⚠️ DISAGREE"
                        st.metric("Model Agreement", agreement_text)
                    
                    st.divider()
                    
                    # Comparison chart
                    fig = go.Figure()
                    models = [selected_model, result['prediction']['best_model_name']]
                    probs = [result['prediction']['churn_probability'], result['prediction']['best_model_probability']]
                    colors = ['#FF6B6B' if p > 0.5 else '#4ECDC4' for p in probs]
                    
                    fig.add_trace(go.Bar(
                        x=models,
                        y=probs,
                        text=[f"{p:.1%}" for p in probs],
                        textposition='auto',
                        marker=dict(color=colors),
                        showlegend=False
                    ))
                    fig.update_layout(
                        title="Churn Probability Comparison",
                        yaxis_title="Probability",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"❌ {result['error']}")
    
    # ==================== TAB 2: BUSINESS IMPACT ====================
    with tab2:
        st.header("💰 Business Impact Analysis")
        
        if st.session_state.current_prediction:
            result = st.session_state.current_prediction
            
            col1, col2 = st.columns([2, 1])
            with col1:
                expected_tenure = st.slider("Expected Customer Lifetime (months):", 12, 60, 24)
            
            # Recalculate with new tenure
            revenue_result = backend.calculate_revenue_loss(
                result['prediction']['churn_probability'],
                result['business_impact']['monthly_charges'],
                expected_tenure
            )
            
            st.divider()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Churn Probability", f"{result['prediction']['churn_probability']:.1%}")
            with col2:
                st.metric("💰 Revenue at Risk", f"${revenue_result['revenue_loss']:.2f}")
            with col3:
                risk_icon = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
                st.metric("Risk Level", f"{risk_icon[revenue_result['risk_level']]} {revenue_result['risk_level']}")
            with col4:
                st.metric("Lifetime Value", f"${revenue_result['lifetime_value']:.2f}")
            
            st.divider()
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['prediction']['churn_probability'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Risk Score (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "lightyellow"},
                        {'range': [60, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': result['prediction']['churn_probability'] * 100
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Revenue Analysis")
            revenue_data = {
                'Metric': ['Monthly Revenue', 'Annual Revenue', 'Lifetime Revenue', 'Revenue at Risk', 'Safe Revenue'],
                'Amount ($)': [
                    revenue_result['monthly_charges'],
                    revenue_result['monthly_charges'] * 12,
                    revenue_result['lifetime_value'],
                    revenue_result['revenue_loss'],
                    revenue_result['lifetime_value'] - revenue_result['revenue_loss']
                ]
            }
            df = pd.DataFrame(revenue_data)
            df['Amount ($)'] = df['Amount ($)'].apply(lambda x: f"${x:.2f}")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("👆 Make a prediction first")
    
    # ==================== TAB 3: EXPLAINABILITY ====================
    with tab3:
        st.header("🔍 Model Explainability")
        
        if st.session_state.current_prediction:
            result = st.session_state.current_prediction
            
            # SHAP Analysis
            st.subheader("🧠 Feature Importance (SHAP Analysis)")
            
            if result['shap_explanation']['success']:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    features_df = pd.DataFrame(result['shap_explanation']['top_features'])
                    fig = go.Figure(go.Bar(
                        x=features_df['shap_value'],
                        y=features_df['name'],
                        orientation='h',
                        marker=dict(
                            color=features_df['shap_value'],
                            colorscale='RdBu',
                            colorbar=dict(title="SHAP Value")
                        ),
                        text=[f"{v:.3f}" for v in features_df['shap_value']],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title="Top 5 Features for This Prediction",
                        xaxis_title="SHAP Value (Contribution to Churn)",
                        yaxis_title="Feature",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.caption("**💡 SHAP Interpretation**")
                    st.markdown("""
**Positive values** ➡️ Increase churn risk  
**Negative values** ⬅️ Decrease churn risk  
**Larger magnitude** = Stronger impact
                    """)
                    st.divider()
                    st.caption("**Impact Levels**")
                    for feat in result['shap_explanation']['top_features']:
                        impact_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
                        st.markdown(f"{impact_color[feat['impact']]} **{feat['name']}**")
            else:
                st.warning(f"⚠️ SHAP analysis unavailable: {result['shap_explanation'].get('error', 'Unknown error')}")
            
            st.divider()
            
            # Global importance
            st.subheader("🌍 Global Feature Importance")
            st.caption("Features that matter most across ALL predictions")
            
            if result['shap_explanation']['success'] and result['shap_explanation'].get('global_importance'):
                global_df = pd.DataFrame(result['shap_explanation']['global_importance'])
                fig = px.bar(
                    global_df,
                    x='importance',
                    y='name',
                    orientation='h',
                    title='Most Important Features Across All Predictions',
                    labels={'importance': 'Average |SHAP Value|', 'name': 'Feature'},
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # LLM Insights
            st.subheader("🤖 AI-Generated Insights & Retention Strategy")
            if result['llm_insights']['success']:
                st.markdown(result['llm_insights']['insights'])
                
                if result['llm_insights'].get('fallback'):
                    st.info("💡 **Tip**: Configure GEMINI_API_KEY in .env for enhanced AI-powered insights")
            else:
                st.warning("⚠️ LLM insights unavailable. Configure Gemini API key in .env file.")
        else:
            st.info("👆 Make a prediction first")
    
    # ==================== TAB 4: MODEL COMPARISON ====================
    with tab4:
        st.header("📈 Model Performance Comparison")
        
        if backend.model_metrics:
            st.subheader("📊 All Metrics")
            metrics_df = pd.DataFrame(backend.model_metrics).T[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].round(3)
            st.dataframe(metrics_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    metrics_df.reset_index().melt(id_vars='index'),
                    x='index',
                    y='value',
                    color='variable',
                    barmode='group',
                    title='Performance Metrics Comparison',
                    labels={'index': 'Model', 'value': 'Score', 'variable': 'Metric'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                roc_data = []
                for model, metrics in backend.model_metrics.items():
                    roc_data.append({'Model': model, 'ROC-AUC': metrics['roc_auc']})
                
                fig = px.bar(
                    roc_data,
                    x='Model',
                    y='ROC-AUC',
                    title='ROC-AUC Comparison',
                    color='ROC-AUC',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Model Comparison Analysis with explanation
            model_analysis = backend.get_model_comparison_analysis()
            if model_analysis['success']:
                st.subheader("🏆 Why This Model is Best")
                st.markdown(model_analysis['analysis']['explanation'])
            
            st.divider()
            
            # ROC Analysis
            st.subheader("📉 ROC-AUC Analysis")
            roc_analysis = backend.get_roc_analysis()
            if roc_analysis['success']:
                st.markdown(roc_analysis['analysis'])
                
                # Plot ROC curves
                fig = go.Figure()
                for model_name, roc_data in roc_analysis['roc_data'].items():
                    fig.add_trace(go.Scatter(
                        x=roc_data['fpr'],
                        y=roc_data['tpr'],
                        mode='lines',
                        name=f"{model_name} (AUC={roc_data['auc']:.3f})",
                        line=dict(width=3)
                    ))
                
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Random (AUC=0.5)',
                    line=dict(dash='dash', color='gray', width=2)
                ))
                
                fig.update_layout(
                    title='ROC Curves - All Models',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No models trained yet")
    
    # ==================== TAB 5: MONITORING ====================
    with tab5:
        st.header("⚠️ Monitoring & Drift Detection")
        
        if len(backend.predictions_history) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", len(backend.predictions_history))
            with col2:
                churn_count = sum(1 for p in backend.predictions_history if p['prediction'] == 1)
                st.metric("Predicted Churn", f"{churn_count}/{len(backend.predictions_history)}")
            with col3:
                churn_rate = churn_count / len(backend.predictions_history) * 100 if backend.predictions_history else 0
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
            
            st.divider()
            
            # Recent predictions
            st.subheader("📈 Recent Predictions")
            recent_preds = backend.predictions_history[-10:]
            pred_data = [{
                'Time': p['timestamp'].split('T')[1][:8] if 'T' in p['timestamp'] else p['timestamp'],
                'Model': p['model'],
                'Probability': f"{p['probability']:.1%}",
                'Prediction': '🚨 CHURN' if p['prediction'] == 1 else '✅ RETAIN'
            } for p in recent_preds]
            
            st.dataframe(pd.DataFrame(pred_data), use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Drift Detection
            st.subheader("🔍 Data Drift Analysis")
            st.caption("Detects if recent predictions differ significantly from training data distribution")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🔍 Check for Drift", use_container_width=True):
                    with st.spinner("Analyzing drift..."):
                        drift_result = backend.detect_drift(backend.predictions_history)
                    
                    if drift_result['success']:
                        if drift_result['drift_detected']:
                            st.warning(drift_result['message'])
                        else:
                            st.success(drift_result['message'])
                    else:
                        st.error(f"❌ Drift analysis failed: {drift_result.get('message', 'Unknown error')}")
            
            with col2:
                st.info("""
**What is Data Drift?**
Data drift occurs when the statistical properties of input features change over time.
This can reduce model accuracy and indicates models may need retraining.

**When to check:** After accumulating 20+ predictions or weekly for production systems.
                """)
            
            st.divider()
            
            # Feedback Loop
            st.subheader("🔄 Provide Feedback & Retrain")
            st.info("Upload actual churn outcomes to retrain models with real-world feedback")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                feedback_file = st.file_uploader("Upload feedback CSV (same format as training data)", type=['csv'])
            
            with col2:
                if feedback_file and st.button("📊 Retrain Models", use_container_width=True, type="primary"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                        tmp.write(feedback_file.getbuffer())
                        feedback_path = tmp.name
                    
                    with st.spinner("Retraining all models..."):
                        retrain_result = backend.retrain_with_feedback(feedback_path)
                        if retrain_result['success']:
                            st.success("✅ Models retrained successfully!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"❌ {retrain_result['error']}")
        else:
            st.info("📊 No predictions yet. Make predictions in the 'Prediction' tab to enable monitoring.")
    
    # ==================== TAB 6: REPORTS ====================
    with tab6:
        st.header("📄 Comprehensive Reports")
        
        if st.session_state.current_prediction:
            result = st.session_state.current_prediction
            
            # Summary Report
            st.subheader("📋 Prediction Summary")
            
            report_text = f"""
## Customer Churn Analysis Report
**Generated:** {result['generated_at']}

---

### 🎯 Prediction Results
- **Selected Model:** {result['prediction']['model_name']}
- **Churn Probability:** {result['prediction']['churn_probability']:.1%}
- **Prediction:** {'🚨 HIGH RISK - CHURN LIKELY' if result['prediction']['churn_prediction'] == 1 else '✅ LOW RISK - LIKELY TO RETAIN'}
- **Best Model:** {result['prediction']['best_model_name']} ({result['prediction']['best_model_probability']:.1%})
- **Model Agreement:** {'✅ Models Agree' if result['prediction']['agreement'] else '⚠️ Models Disagree'}

---

### 💰 Business Impact
- **Monthly Charges:** ${result['business_impact']['monthly_charges']:.2f}
- **Revenue at Risk:** ${result['business_impact']['revenue_loss']:.2f}
- **Customer Lifetime Value:** ${result['business_impact']['lifetime_value']:.2f}
- **Risk Level:** {result['business_impact']['risk_level']}
- **Loss Percentage:** {result['business_impact']['loss_percentage']:.1f}% of total lifetime value

---

### 🔍 Top Risk Factors (SHAP Analysis)
"""
            
            if result['shap_explanation']['success']:
                report_text += "\n**Features Contributing to Churn Risk:**\n\n"
                for i, feat in enumerate(result['shap_explanation']['top_features'][:5], 1):
                    direction = "increases" if feat['shap_value'] > 0 else "decreases"
                    report_text += f"{i}. **{feat['name']}** ({feat['impact']} Impact)\n"
                    report_text += f"   - SHAP Value: {feat['shap_value']:.4f} ({direction} churn risk)\n"
                    report_text += f"   - Customer Value: {feat['feature_value']:.2f}\n\n"
            else:
                report_text += "\n*SHAP analysis unavailable*\n\n"
            
            report_text += "---\n\n"
            report_text += "### 📊 Model Performance\n\n"
            if backend.model_metrics.get(result['prediction']['model_name']):
                metrics = backend.model_metrics[result['prediction']['model_name']]
                report_text += f"""
- **Accuracy:** {metrics['accuracy']:.3f}
- **Precision:** {metrics['precision']:.3f}
- **Recall:** {metrics['recall']:.3f}
- **F1-Score:** {metrics['f1']:.3f}
- **ROC-AUC:** {metrics['roc_auc']:.3f}
"""
            
            st.markdown(report_text)
            
            st.divider()
            
            # Business Insights Section
            if result['llm_insights']['success']:
                st.subheader("🤖 Business Insights & Recommendations")
                st.markdown(result['llm_insights']['insights'])
            
            st.divider()
            
            # Export Report
            st.subheader("💾 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                json_str = json.dumps(result, indent=2, default=str)
                st.download_button(
                    "📥 Download Full Report (JSON)",
                    json_str,
                    file_name=f"churn_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                metrics_json = pd.DataFrame(backend.model_metrics).T.to_json(indent=2)
                st.download_button(
                    "📥 Download Model Metrics",
                    metrics_json,
                    file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                # Export as text report
                text_report = f"""
CUSTOMER CHURN ANALYSIS REPORT
{'='*60}
Generated: {result['generated_at']}

PREDICTION RESULTS
{'-'*60}
Selected Model: {result['prediction']['model_name']}
Churn Probability: {result['prediction']['churn_probability']:.1%}
Prediction: {'HIGH RISK - CHURN LIKELY' if result['prediction']['churn_prediction'] == 1 else 'LOW RISK - LIKELY TO RETAIN'}
Best Model: {result['prediction']['best_model_name']} ({result['prediction']['best_model_probability']:.1%})
Model Agreement: {'Models Agree' if result['prediction']['agreement'] else 'Models Disagree'}

BUSINESS IMPACT
{'-'*60}
Monthly Charges: ${result['business_impact']['monthly_charges']:.2f}
Revenue at Risk: ${result['business_impact']['revenue_loss']:.2f}
Customer Lifetime Value: ${result['business_impact']['lifetime_value']:.2f}
Risk Level: {result['business_impact']['risk_level']}
Loss Percentage: {result['business_impact']['loss_percentage']:.1f}%

TOP RISK FACTORS
{'-'*60}
"""
                if result['shap_explanation']['success']:
                    for i, feat in enumerate(result['shap_explanation']['top_features'][:5], 1):
                        direction = "increases" if feat['shap_value'] > 0 else "decreases"
                        text_report += f"{i}. {feat['name']} ({feat['impact']} Impact)\n"
                        text_report += f"   SHAP Value: {feat['shap_value']:.4f} ({direction} churn risk)\n"
                        text_report += f"   Customer Value: {feat['feature_value']:.2f}\n\n"
                
                text_report += f"""
MODEL PERFORMANCE
{'-'*60}
"""
                if backend.model_metrics.get(result['prediction']['model_name']):
                    metrics = backend.model_metrics[result['prediction']['model_name']]
                    text_report += f"""Accuracy: {metrics['accuracy']:.3f}
Precision: {metrics['precision']:.3f}
Recall: {metrics['recall']:.3f}
F1-Score: {metrics['f1']:.3f}
ROC-AUC: {metrics['roc_auc']:.3f}

{'='*60}
End of Report
"""
                
                st.download_button(
                    "📥 Download Text Report",
                    text_report,
                    file_name=f"churn_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("👆 Make a prediction first to generate reports")

# ==================== FOOTER ====================
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    f"🎯 Advanced Churn Prediction Platform | v2.0 | Powered by ML + SHAP + Gemini AI<br>"
    f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</div>",
    unsafe_allow_html=True
)