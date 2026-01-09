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
import io

load_dotenv()

st.set_page_config(
    page_title="🎯 Churn Prediction Platform",
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
    .success-box {
        background-color: #d1ecf1;
        border: 1px solid #0dcaf0;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        color: #000000;
    }
    .success-box h3, .success-box p, .success-box strong {
        color: #000000 !important;
    }    
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        color: #000000;
    }
    .warning-box h3, .warning-box p, .warning-box strong {
        color: #000000 !important;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        color: #000000;
    }
    .danger-box h3, .danger-box p, .danger-box strong {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'backend' not in st.session_state:
    st.session_state.backend = ChurnPredictionBackend()
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = st.session_state.backend.models.get('Ensemble') is not None
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'batch_predictions' not in st.session_state:
    st.session_state.batch_predictions = None
if 'feature_types' not in st.session_state:
    st.session_state.feature_types = st.session_state.backend.feature_types

backend = st.session_state.backend

# Sidebar
with st.sidebar:
    st.title("🎯 Churn Prediction")
    st.markdown("**ML Platform v2.0**")
    st.divider()
    
    st.subheader("📊 Configuration")
    try:
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
        if GEMINI_API_KEY and not GEMINI_API_KEY.startswith('YOUR_'):
            st.success("✅ Gemini API Active")
        else:
            st.warning("⚠️ Gemini API Not Configured")
            with st.expander("Configure Gemini API"):
                st.info("Set GEMINI_API_KEY in .env file for AI insights")
    except Exception as e:
        st.error(f"⚠️ Gemini API Error: {str(e)}")
    
    st.divider()
    
    # Model Status
    if st.session_state.trained_models and backend.model_metrics:
        st.subheader("🤖 Model Status")
        best_model = backend._get_best_model()
        st.success(f"**Best Model:** {best_model}")
        
        with st.expander("All Models Performance"):
            for model_name, metrics in backend.model_metrics.items():
                is_best = "⭐" if model_name == best_model else ""
                st.markdown(f"**{model_name}** {is_best}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"F1: {metrics['f1']:.3f}")
                with col2:
                    st.caption(f"Recall: {metrics['recall']:.3f}")
                with col3:
                    st.caption(f"AUC: {metrics['roc_auc']:.3f}")
                st.divider()
    
    st.divider()
    
    # Feedback Upload in Sidebar
    st.subheader("📥 Upload Feedback Data")
    st.caption("Retrain models with actual churn outcomes")
    
    feedback_file = st.file_uploader(
        "Upload Feedback CSV",
        type=['csv'],
        key='sidebar_feedback',
        help="Upload CSV with same format as training data including actual churn outcomes"
    )
    
    if feedback_file:
        with st.expander("Preview Feedback Data"):
            feedback_preview = pd.read_csv(feedback_file)
            st.dataframe(feedback_preview.head(5), use_container_width=True)
            st.caption(f"Rows: {len(feedback_preview)} | Columns: {len(feedback_preview.columns)}")
        
        if st.button("🔄 Retrain All Models", use_container_width=True, type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(feedback_file.getbuffer())
                feedback_path = tmp.name
            
            with st.spinner("🔄 Retraining all models with new data..."):
                retrain_result = backend.retrain_with_feedback(feedback_path)
                if retrain_result['success']:
                    st.success("✅ Models retrained successfully!")
                    st.balloons()
                    for log_line in retrain_result.get('log', []):
                        st.caption(log_line)
                    st.rerun()
                else:
                    st.error(f"❌ {retrain_result['error']}")
    
    st.divider()
    
    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset", help="Clear all data", use_container_width=True):
            st.session_state.trained_models = False
            st.session_state.current_prediction = None
            st.session_state.batch_predictions = None
            st.rerun()
    
    with col2:
        if st.button("ℹ️ About", use_container_width=True):
            st.info("""
            **Features:**
            - 5 ML Models
            - Single & Batch Predictions
            - SHAP Explainability
            - Business Impact Analysis
            - Real-time Retraining
            - Drift Detection
            - Comprehensive Reports
            """)

# Main Content
st.title("🎯 Customer Churn Prediction Platform")
st.markdown("*Enterprise-Grade Churn Prediction with AI-Powered Insights*")

if not st.session_state.trained_models or backend.training_data is None:
    st.header("🚀 Step 1: Train Your Models")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload Training Data")
        uploaded_file = st.file_uploader(
            "Upload CSV (must contain 'churn' or 'Churn' column)",
            type=['csv'],
            help="Dataset should contain 'churn' column and at least 100 rows"
        )
        
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(uploaded_file.getbuffer())
                csv_path = tmp.name
            
            df_preview = pd.read_csv(csv_path)
            st.caption(f"📊 Rows: {len(df_preview)} | Columns: {len(df_preview.columns)}")
            
            with st.expander("👀 Preview Training Data"):
                st.dataframe(df_preview.head(20), use_container_width=True)
            
            col_check1, col_check2 = st.columns(2)
            with col_check1:
                churn_col = None
                for col in ['churn', 'Churn', 'CHURN', 'ChurnRisk']:
                    if col in df_preview.columns:
                        churn_col = col
                        break
                
                if churn_col:
                    st.success(f"✅ Target column '{churn_col}' found")
                else:
                    st.error("❌ No 'churn' column found")
            
            with col_check2:
                if len(df_preview) >= 100:
                    st.success(f"✅ Sufficient data ({len(df_preview)} rows)")
                else:
                    st.error(f"❌ Need at least 100 rows (found {len(df_preview)})")
            
            st.divider()
            
            if st.button("🚀 Train All Models (LR, RF, XGB, GB, Ensemble)", 
                        use_container_width=True, 
                        type="primary",
                        disabled=churn_col is None or len(df_preview) < 100):
                with st.spinner("🔄 Training 5 optimized models... This may take a few minutes."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Training Logistic Regression...")
                    progress_bar.progress(20)
                    
                    result = backend.train_models(csv_path)
                    
                    progress_bar.progress(100)
                    
                    if result['success']:
                        st.session_state.trained_models = True
                        st.session_state.feature_types = backend.feature_types
                        st.balloons()
                        st.success("✅ All models trained successfully!")
                        
                        with st.expander("📋 Training Summary"):
                            for log_line in result['log']:
                                st.caption(log_line)
                        
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")
    
    with col2:
        st.subheader("📋 Requirements")
        st.info("""
        **Required Columns:**
        - **churn**: Target variable (Yes/No or 1/0)
        - **Numeric**: tenure, charges, etc.
        - **Categorical**: contract, service, etc.
        
        **Data Quality:**
        - Minimum: 100 rows
        - Recommended: 500+ rows
        - Clean data preferred
        
        **Supported Formats:**
        - CSV files only
        - UTF-8 encoding
        """)
        
        st.divider()
        
        st.subheader("🎯 What You'll Get")
        st.markdown("""
        - ✅ 5 ML Models trained
        - ✅ Single customer predictions
        - ✅ Batch predictions (CSV)
        - ✅ SHAP explanations
        - ✅ Business impact analysis
        - ✅ Model comparison
        - ✅ Drift detection
        - ✅ Export capabilities
        """)

else:
    # Main Prediction Interface
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Single Prediction",
        "📊 Batch Predictions",
        "💰 Business Impact",
        "🔍 Explainability",
        "📈 Model Comparison",
        "🔔 Monitoring",
        "📄 Reports"
    ])
    
    with tab1:
        st.header("🎯 Single Customer Prediction")
        
        col_select, col_info = st.columns([1, 2])
        with col_select:
            model_options = [k for k, v in backend.models.items() if v is not None]
            if 'Ensemble' in model_options:
                default_idx = model_options.index('Ensemble')
            else:
                default_idx = 0
            selected_model = st.selectbox(
                "Select Model:",
                model_options,
                index=default_idx,
                help="Choose which model to use for prediction"
            )
        with col_info:
            best_model = backend._get_best_model()
            st.info(f"⭐ **Recommended Model:** {best_model} (Highest F1-Score)")
        
        st.divider()
        st.subheader("📝 Enter Customer Information")
        
        input_data = {}
        
        if backend.feature_columns:
            numeric_cols = [c for c in backend.feature_columns if backend.feature_types.get(c) == 'numeric']
            categorical_cols = [c for c in backend.feature_columns if backend.feature_types.get(c) == 'categorical']
            
            # Numeric Features
            if numeric_cols:
                st.markdown("### 🔢 Numeric Features")
                cols = st.columns(3)
                for idx, col in enumerate(numeric_cols[:9]):
                    with cols[idx % 3]:
                        if 'tenure' in col.lower():
                            input_data[col] = st.slider(col, 0, 72, 24, help="Customer tenure in months")
                        elif 'charge' in col.lower() or 'cost' in col.lower():
                            input_data[col] = st.number_input(col, 0.0, 500.0, 65.0, help="Monthly charges")
                        else:
                            input_data[col] = st.number_input(col, value=0.0)
                
                if len(numeric_cols) > 9:
                    with st.expander("➕ More Numeric Features"):
                        cols2 = st.columns(3)
                        for idx, col in enumerate(numeric_cols[9:]):
                            with cols2[idx % 3]:
                                input_data[col] = st.number_input(col, value=0.0, key=f"num_{col}")
            
            # Categorical Features
            if categorical_cols:
                st.markdown("### 📑 Categorical Features")
                cols = st.columns(3)
                for idx, col in enumerate(categorical_cols[:6]):
                    with cols[idx % 3]:
                        unique_vals = backend.training_data[col].unique().tolist()
                        input_data[col] = st.selectbox(col, unique_vals, key=f"cat_{col}")
                
                if len(categorical_cols) > 6:
                    with st.expander("➕ More Categorical Features"):
                        cols2 = st.columns(3)
                        for idx, col in enumerate(categorical_cols[6:]):
                            with cols2[idx % 3]:
                                unique_vals = backend.training_data[col].unique().tolist()
                                input_data[col] = st.selectbox(col, unique_vals, key=f"extra_{col}")
        
        st.divider()
        
        col_pred, col_space = st.columns([1, 2])
        with col_pred:
            if st.button("🎯 Generate Comprehensive Prediction & Report", 
                        use_container_width=True, 
                        type="primary"):
                with st.spinner("🔄 Generating comprehensive analysis..."):
                    result = backend.generate_comprehensive_report(input_data, selected_model)
                    
                    if result['success']:
                        st.session_state.current_prediction = result
                        st.success("✅ Prediction complete!")
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")
        
        # Display Results
        if st.session_state.current_prediction:
            result = st.session_state.current_prediction
            
            st.divider()
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                churn_status = "🔴 HIGH RISK" if result['prediction']['churn_prediction'] == 1 else "🟢 LOW RISK"
                st.metric(
                    f"{selected_model}",
                    churn_status,
                    f"{result['prediction']['churn_probability']:.1%}"
                )
            
            with col2:
                best_status = "🔴 HIGH RISK" if result['prediction']['best_model_prediction'] == 1 else "🟢 LOW RISK"
                st.metric(
                    f"{result['prediction']['best_model_name']}",
                    best_status,
                    f"{result['prediction']['best_model_probability']:.1%}"
                )
            
            with col3:
                agreement_icon = "✅" if result['prediction']['agreement'] else "⚠️"
                agreement_text = f"{agreement_icon} {'AGREE' if result['prediction']['agreement'] else 'DISAGREE'}"
                st.metric("Model Agreement", agreement_text)
            
            with col4:
                all_preds = result['prediction'].get('all_model_predictions', {})
                if all_preds:
                    consensus = sum(p['prediction'] for p in all_preds.values()) / len(all_preds)
                    st.metric("Consensus", f"{consensus:.0%}")
            
            st.divider()
            
            # All Model Predictions
            if all_preds:
                st.subheader("🤖 All Model Predictions")
                models = list(all_preds.keys())
                probs = [all_preds[m]['probability'] for m in models]
                colors = ['#FF6B6B' if p > 0.5 else '#4ECDC4' for p in probs]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=models,
                    y=probs,
                    text=[f"{p:.1%}" for p in probs],
                    textposition='auto',
                    marker=dict(color=colors),
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
                ))
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                             annotation_text="50% Threshold")
                fig.update_layout(
                    title="Churn Probability Across All Models",
                    xaxis_title="Model",
                    yaxis_title="Probability",
                    height=400,
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📊 Batch Predictions")
        st.markdown("*Upload a CSV file to predict churn for multiple customers at once*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📤 Upload Customer Data")
            batch_file = st.file_uploader(
                "Upload CSV with customer data (without churn column)",
                type=['csv'],
                key='batch_upload',
                help="Upload CSV with same features as training data (excluding the churn column)"
            )
            
            if batch_file:
                batch_df = pd.read_csv(batch_file)
                st.caption(f"📊 {len(batch_df)} customers | {len(batch_df.columns)} features")
                
                with st.expander("👀 Preview Data"):
                    st.dataframe(batch_df.head(10), use_container_width=True)
                
                col_model, col_btn = st.columns([1, 1])
                with col_model:
                    batch_model = st.selectbox(
                        "Select Model for Batch Prediction:",
                        [k for k, v in backend.models.items() if v is not None],
                        index=0 if 'Ensemble' not in backend.models or backend.models['Ensemble'] is None else list(backend.models.keys()).index('Ensemble'),
                        key='batch_model_select'
                    )
                
                with col_btn:
                    st.write("")  # Spacing
                    st.write("")  # Spacing
                    if st.button("🚀 Predict All Customers", use_container_width=True, type="primary"):
                        with st.spinner(f"🔄 Predicting churn for {len(batch_df)} customers..."):
                            progress_bar = st.progress(0)
                            batch_results = []
                            
                            for idx, row in batch_df.iterrows():
                                input_dict = row.to_dict()
                                pred_result = backend.predict(input_dict, batch_model)
                                
                                if pred_result['success']:
                                    monthly_charges = input_dict.get('MonthlyCharges', 65) if isinstance(input_dict.get('MonthlyCharges'), (int, float)) else 65
                                    revenue_analysis = backend.calculate_revenue_loss(
                                        pred_result['churn_probability'],
                                        monthly_charges
                                    )
                                    
                                    batch_results.append({
                                        'Customer_ID': input_dict.get('customerID', f'Customer_{idx+1}'),
                                        'Churn_Probability': pred_result['churn_probability'],
                                        'Churn_Prediction': 'Churn' if pred_result['churn_prediction'] == 1 else 'Retain',
                                        'Risk_Level': revenue_analysis['risk_level'],
                                        'Revenue_at_Risk': revenue_analysis['revenue_loss'],
                                        'Monthly_Charges': monthly_charges,
                                        'Model_Used': batch_model
                                    })
                                
                                progress_bar.progress((idx + 1) / len(batch_df))
                            
                            st.session_state.batch_predictions = pd.DataFrame(batch_results)
                            st.success(f"✅ Successfully predicted churn for {len(batch_results)} customers!")
        
        with col2:
            st.subheader("📋 Instructions")
            st.info("""
            **How to use:**
            1. Prepare CSV with customer data
            2. Include all features used in training
            3. Exclude the 'churn' column
            4. Upload and click Predict
            
            **Output includes:**
            - Churn probability
            - Risk classification
            - Revenue impact
            - Actionable insights
            """)
        
        # Display Batch Results
        if st.session_state.batch_predictions is not None:
            batch_df = st.session_state.batch_predictions
            
            st.divider()
            st.subheader("📊 Batch Prediction Results")
            
            # Summary Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_customers = len(batch_df)
                st.metric("Total Customers", total_customers)
            
            with col2:
                churn_count = len(batch_df[batch_df['Churn_Prediction'] == 'Churn'])
                churn_rate = (churn_count / total_customers * 100) if total_customers > 0 else 0
                st.metric("Predicted Churners", f"{churn_count} ({churn_rate:.1f}%)")
            
            with col3:
                total_revenue_risk = batch_df['Revenue_at_Risk'].sum()
                st.metric("Total Revenue at Risk", f"${total_revenue_risk:,.2f}")
            
            with col4:
                high_risk = len(batch_df[batch_df['Risk_Level'] == 'High'])
                st.metric("High Risk Customers", high_risk)
            
            st.divider()
            
            # Risk Distribution
            col1, col2 = st.columns(2)
            
            with col1:
                risk_counts = batch_df['Risk_Level'].value_counts()
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title='Risk Level Distribution',
                    color_discrete_map={'Low': '#4ECDC4', 'Medium': '#FFE66D', 'High': '#FF6B6B'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                prob_ranges = pd.cut(batch_df['Churn_Probability'], 
                                    bins=[0, 0.3, 0.6, 1.0], 
                                    labels=['Low (0-30%)', 'Medium (30-60%)', 'High (60-100%)'])
                prob_counts = prob_ranges.value_counts().sort_index()
                
                fig = go.Figure(data=[
                    go.Bar(x=prob_counts.index, y=prob_counts.values,
                          marker_color=['#4ECDC4', '#FFE66D', '#FF6B6B'])
                ])
                fig.update_layout(title='Churn Probability Distribution',
                                xaxis_title='Probability Range',
                                yaxis_title='Number of Customers')
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Detailed Results Table
            st.subheader("📋 Detailed Results")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_risk = st.multiselect(
                    "Filter by Risk Level:",
                    options=['Low', 'Medium', 'High'],
                    default=['Low', 'Medium', 'High']
                )
            with col2:
                filter_prediction = st.multiselect(
                    "Filter by Prediction:",
                    options=['Churn', 'Retain'],
                    default=['Churn', 'Retain']
                )
            with col3:
                sort_by = st.selectbox(
                    "Sort by:",
                    options=['Churn_Probability', 'Revenue_at_Risk', 'Customer_ID'],
                    index=0
                )
            
            # Apply filters
            filtered_df = batch_df[
                (batch_df['Risk_Level'].isin(filter_risk)) &
                (batch_df['Churn_Prediction'].isin(filter_prediction))
            ].sort_values(by=sort_by, ascending=False)
            
            # Format for display
            display_df = filtered_df.copy()
            display_df['Churn_Probability'] = display_df['Churn_Probability'].apply(lambda x: f"{x:.1%}")
            display_df['Revenue_at_Risk'] = display_df['Revenue_at_Risk'].apply(lambda x: f"${x:,.2f}")
            display_df['Monthly_Charges'] = display_df['Monthly_Charges'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            st.caption(f"Showing {len(filtered_df)} of {len(batch_df)} customers")
            
            # Export Options
            st.divider()
            st.subheader("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Full Results (CSV)",
                    csv,
                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                high_risk_df = batch_df[batch_df['Risk_Level'] == 'High']
                high_risk_csv = high_risk_df.to_csv(index=False)
                st.download_button(
                    "🔴 Download High Risk Only (CSV)",
                    high_risk_csv,
                    file_name=f"high_risk_customers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                json_str = batch_df.to_json(orient='records', indent=2)
                st.download_button(
                    "📄 Download as JSON",
                    json_str,
                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    with tab3:
        st.header("💰 Business Impact Analysis")
        
        if st.session_state.current_prediction:
            result = st.session_state.current_prediction
            
            col1, col2 = st.columns([2, 1])
            with col1:
                expected_tenure = st.slider("Expected Customer Lifetime (months):", 12, 60, 24)
            with col2:
                st.info("💡 Average telecom customer lifetime: 24-36 months")
            
            revenue_result = backend.calculate_revenue_loss(
                result['prediction']['churn_probability'],
                result['business_impact']['monthly_charges'],
                expected_tenure
            )
            
            st.divider()
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💳 Monthly Charges", f"${revenue_result['monthly_charges']:.2f}")
            with col2:
                st.metric("💰 Lifetime Value", f"${revenue_result['lifetime_value']:,.2f}")
            with col3:
                st.metric("⚠️ Revenue at Risk", f"${revenue_result['revenue_loss']:,.2f}")
            with col4:
                risk_icon = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
                st.metric("📊 Risk Level", f"{risk_icon[revenue_result['risk_level']]} {revenue_result['risk_level']}")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=result['prediction']['churn_probability'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Risk Score (%)"},
                    delta={'reference': 50},
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
            
            with col2:
                # Revenue Breakdown
                revenue_data = {
                    'Category': ['Safe Revenue', 'Revenue at Risk'],
                    'Amount': [
                        revenue_result['lifetime_value'] - revenue_result['revenue_loss'],
                        revenue_result['revenue_loss']
                    ]
                }
                fig = px.pie(
                    revenue_data,
                    values='Amount',
                    names='Category',
                    title='Revenue Distribution',
                    color='Category',
                    color_discrete_map={'Safe Revenue': '#4ECDC4', 'Revenue at Risk': '#FF6B6B'}
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Detailed Revenue Analysis
            st.subheader("📊 Detailed Revenue Analysis")
            revenue_breakdown = pd.DataFrame({
                'Metric': [
                    'Monthly Revenue',
                    'Quarterly Revenue',
                    'Annual Revenue',
                    'Total Lifetime Revenue',
                    'Revenue at Risk',
                    'Protected Revenue',
                    'Loss Percentage'
                ],
                'Amount': [
                    f"${revenue_result['monthly_charges']:.2f}",
                    f"${revenue_result['monthly_charges'] * 3:.2f}",
                    f"${revenue_result['monthly_charges'] * 12:.2f}",
                    f"${revenue_result['lifetime_value']:,.2f}",
                    f"${revenue_result['revenue_loss']:,.2f}",
                    f"${revenue_result['lifetime_value'] - revenue_result['revenue_loss']:,.2f}",
                    f"{revenue_result['loss_percentage']:.1f}%"
                ]
            })
            st.dataframe(revenue_breakdown, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # ROI Calculation for Retention
            st.subheader("💡 Retention Strategy ROI")
            
            col1, col2 = st.columns(2)
            with col1:
                retention_cost = st.number_input(
                    "Estimated Retention Campaign Cost ($):",
                    min_value=0.0,
                    max_value=1000.0,
                    value=100.0,
                    step=10.0
                )
            with col2:
                retention_success_rate = st.slider(
                    "Expected Success Rate (%):",
                    min_value=0,
                    max_value=100,
                    value=60,
                    step=5
                )
            
            expected_saved = revenue_result['revenue_loss'] * (retention_success_rate / 100)
            net_benefit = expected_saved - retention_cost
            roi = ((expected_saved - retention_cost) / retention_cost * 100) if retention_cost > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Expected Revenue Saved", f"${expected_saved:,.2f}")
            with col2:
                st.metric("📈 Net Benefit", f"${net_benefit:,.2f}")
            with col3:
                st.metric("🎯 ROI", f"{roi:.0f}%")
            
            if roi > 200:
                st.success(f"✅ Excellent ROI! Retention campaign highly recommended.")
            elif roi > 100:
                st.info(f"👍 Good ROI. Retention campaign is worthwhile.")
            elif roi > 0:
                st.warning(f"⚠️ Moderate ROI. Consider retention for high-value customers.")
            else:
                st.error(f"❌ Negative ROI. Review retention strategy or customer value.")
        
        else:
            st.info("📊 Make a prediction first to see business impact analysis")
    
    with tab4:
        st.header("🔍 Model Explainability (SHAP Analysis)")
        
        if st.session_state.current_prediction:
            result = st.session_state.current_prediction
            
            st.markdown("### Understanding What Drives Churn Risk")
            
            if result['shap_explanation']['success']:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("🎯 Top 5 Features for This Prediction")
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
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<br>Feature Value: %{customdata}<extra></extra>',
                        customdata=features_df['feature_value']
                    ))
                    fig.update_layout(
                        title="Feature Impact on Churn Prediction",
                        xaxis_title="SHAP Value (Contribution to Churn Risk)",
                        yaxis_title="Feature",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 📖 How to Read")
                    st.info("""
                    **SHAP Values Explained:**
                    
                    🔴 **Positive values**: 
                    Increase churn risk
                    
                    🔵 **Negative values**: 
                    Decrease churn risk
                    
                    📏 **Magnitude**: 
                    Larger = Stronger impact
                    """)
                    
                    st.divider()
                    
                    st.markdown("### 🎚️ Impact Levels")
                    for feat in result['shap_explanation']['top_features']:
                        impact_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
                        st.markdown(f"{impact_color.get(feat['impact'], '⚪')} **{feat['name']}**: {feat['impact']}")
                
                st.divider()
                
                # Feature Details Table
                st.subheader("📋 Detailed Feature Analysis")
                
                feature_details = []
                for feat in result['shap_explanation']['top_features']:
                    direction = "Increases" if feat['shap_value'] > 0 else "Decreases"
                    feature_details.append({
                        'Feature': feat['name'],
                        'Customer Value': f"{feat['feature_value']:.2f}",
                        'SHAP Value': f"{feat['shap_value']:.4f}",
                        'Effect': f"{direction} churn risk",
                        'Impact Level': feat['impact']
                    })
                
                st.dataframe(pd.DataFrame(feature_details), use_container_width=True, hide_index=True)
                
                st.divider()
                
                # Global Feature Importance
                st.subheader("🌍 Global Feature Importance")
                st.caption("Features that matter most across ALL predictions in the training data")
                
                if result['shap_explanation'].get('global_importance'):
                    global_df = pd.DataFrame(result['shap_explanation']['global_importance'])
                    
                    fig = px.bar(
                        global_df,
                        x='importance',
                        y='name',
                        orientation='h',
                        title='Most Important Features (Global Analysis)',
                        labels={'importance': 'Average |SHAP Value|', 'name': 'Feature'},
                        color='importance',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("""
                    💡 **Global vs Local Importance:**
                    - **Global**: Most important features across all customers
                    - **Local**: Most important features for THIS specific customer
                    """)
            else:
                st.warning(f"⚠️ SHAP analysis unavailable: {result['shap_explanation'].get('error', 'Unknown error')}")
            
            st.divider()
            
            # AI-Generated Insights
            st.subheader("🤖 AI-Generated Insights & Retention Strategy")
            
            if result['llm_insights']['success']:
                if result['llm_insights'].get('fallback'):
                    st.info("💡 Using rule-based insights (Gemini API not configured)")
                else:
                    st.success("✨ AI-powered insights from Gemini")
                
                st.markdown(result['llm_insights']['insights'])
            else:
                st.warning("⚠️ AI insights unavailable. Configure Gemini API key for enhanced insights.")
        
        else:
            st.info("🔍 Make a prediction first to see explainability analysis")
    
    with tab5:
        st.header("📈 Model Performance Comparison")
        
        if backend.model_metrics:
            # Best Model Highlight
            best_model = backend._get_best_model()
            st.success(f"⭐ **Best Overall Model: {best_model}** (Highest F1-Score)")
            
            st.divider()
            
            # Metrics Table
            st.subheader("📊 All Model Metrics")
            metrics_df = pd.DataFrame(backend.model_metrics).T[['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'threshold']].round(3)
            
            # Highlight best model
            def highlight_best(s):
                if s.name in backend.model_metrics:
                    if s.name == best_model:
                        return ['background-color: #d4edda; color: #000000; font-weight: bold'] * len(s)
                return ['color: #ffffff'] * len(s)
            
            styled_df = metrics_df.style.apply(highlight_best, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            st.divider()
            
            # Visual Comparisons
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Metrics Comparison")
                fig = px.bar(
                    metrics_df.reset_index().melt(id_vars='index', value_vars=['accuracy', 'precision', 'recall', 'f1']),
                    x='index',
                    y='value',
                    color='variable',
                    barmode='group',
                    title='Performance Metrics by Model',
                    labels={'index': 'Model', 'value': 'Score', 'variable': 'Metric'}
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 ROC-AUC Scores")
                roc_data = pd.DataFrame({
                    'Model': metrics_df.index,
                    'ROC-AUC': metrics_df['roc_auc'].values
                }).sort_values('ROC-AUC', ascending=False)
                
                fig = px.bar(
                    roc_data,
                    x='Model',
                    y='ROC-AUC',
                    title='ROC-AUC Comparison',
                    color='ROC-AUC',
                    color_continuous_scale='Viridis',
                    text='ROC-AUC'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Model Analysis
            st.subheader("📝 Why This Model is Best")
            model_analysis = backend.get_model_comparison_analysis()
            if model_analysis['success']:
                st.markdown(model_analysis['analysis']['explanation'])
            
            st.divider()
            
            # ROC Curves
            st.subheader("📈 ROC Curves - Model Discrimination Ability")
            
            roc_analysis = backend.get_roc_analysis()
            if roc_analysis['success']:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = go.Figure()
                    
                    # Add ROC curves for all models
                    for model_name, roc_data in roc_analysis['roc_data'].items():
                        fig.add_trace(go.Scatter(
                            x=roc_data['fpr'],
                            y=roc_data['tpr'],
                            mode='lines',
                            name=f"{model_name} (AUC={roc_data['auc']:.3f})",
                            line=dict(width=3),
                            hovertemplate='<b>%{fullData.name}</b><br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
                        ))
                    
                    # Add diagonal reference line
                    fig.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        name='Random Classifier (AUC=0.5)',
                        line=dict(dash='dash', color='gray', width=2)
                    ))
                    
                    fig.update_layout(
                        title='ROC Curves - All Models',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate (Recall)',
                        height=500,
                        hovermode='closest'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 📖 Understanding ROC-AUC")
                    st.info("""
                    **ROC-AUC Scale:**
                    - 0.5 = Random guess
                    - 0.7-0.8 = Good
                    - 0.8-0.9 = Excellent
                    - 0.9+ = Outstanding
                    
                    **What it means:**
                    Probability that the model ranks a random churner higher than a random non-churner.
                    
                    **Higher = Better** discrimination ability
                    """)
                
                st.markdown(roc_analysis['analysis'])
            
            st.divider()
            
            # Threshold Analysis
            st.subheader("🎚️ Optimized Thresholds")
            st.caption("Custom thresholds for maximum recall (catching churners)")
            
            threshold_data = []
            for model_name, metrics in backend.model_metrics.items():
                threshold_data.append({
                    'Model': model_name,
                    'Threshold': metrics.get('threshold', 0.5),
                    'Recall': metrics['recall'],
                    'Precision': metrics['precision'],
                    'F1-Score': metrics['f1']
                })
            
            threshold_df = pd.DataFrame(threshold_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=threshold_df['Model'],
                y=threshold_df['Threshold'],
                mode='markers+lines',
                name='Threshold',
                marker=dict(size=12),
                line=dict(width=2)
            ))
            fig.update_layout(
                title='Custom Thresholds by Model',
                xaxis_title='Model',
                yaxis_title='Threshold',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(threshold_df, use_container_width=True, hide_index=True)
            
            st.info("""
            💡 **Why Custom Thresholds?**
            - Default threshold (0.5) may miss churners
            - Lower thresholds (0.3-0.4) catch more churners
            - Trade-off: More false positives, but fewer missed churners
            - Better to contact extra customers than miss real churners
            """)
        
        else:
            st.info("📊 No models trained yet")
    
    with tab6:
        st.header("🔔 Monitoring & Drift Detection")
        
        if len(backend.predictions_history) > 0:
            # Summary Metrics
            st.subheader("📊 Prediction Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📈 Total Predictions", len(backend.predictions_history))
            
            with col2:
                churn_count = sum(1 for p in backend.predictions_history if p['prediction'] == 1)
                st.metric("🔴 Predicted Churners", churn_count)
            
            with col3:
                churn_rate = (churn_count / len(backend.predictions_history) * 100) if backend.predictions_history else 0
                st.metric("📊 Churn Rate", f"{churn_rate:.1f}%")
            
            with col4:
                avg_prob = np.mean([p['probability'] for p in backend.predictions_history])
                st.metric("📉 Avg Churn Probability", f"{avg_prob:.1%}")
            
            st.divider()
            
            # Prediction Timeline
            st.subheader("📈 Prediction Timeline")
            
            timeline_data = []
            for p in backend.predictions_history:
                timeline_data.append({
                    'Timestamp': p['timestamp'],
                    'Probability': p['probability'],
                    'Prediction': 'Churn' if p['prediction'] == 1 else 'Retain',
                    'Model': p['model']
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df['Timestamp'] = pd.to_datetime(timeline_df['Timestamp'])
            
            fig = px.scatter(
                timeline_df,
                x='Timestamp',
                y='Probability',
                color='Prediction',
                symbol='Model',
                title='Churn Probability Over Time',
                color_discrete_map={'Churn': '#FF6B6B', 'Retain': '#4ECDC4'}
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50% Threshold")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Recent Predictions Table
            st.subheader("📋 Recent Predictions")
            
            display_count = st.slider("Number of recent predictions to show:", 5, 50, 10)
            
            recent_preds = backend.predictions_history[-display_count:]
            pred_data = []
            for p in recent_preds:
                timestamp_str = p['timestamp']
                if 'T' in timestamp_str:
                    date_part, time_part = timestamp_str.split('T')
                    time_str = time_part[:8]
                else:
                    date_part = timestamp_str[:10]
                    time_str = timestamp_str[11:19] if len(timestamp_str) > 10 else '00:00:00'
                
                pred_data.append({
                    'Date': date_part,
                    'Time': time_str,
                    'Model': p['model'],
                    'Probability': f"{p['probability']:.1%}",
                    'Prediction': '🔴 CHURN' if p['prediction'] == 1 else '🟢 RETAIN'
                })
            
            st.dataframe(pd.DataFrame(pred_data), use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Data Drift Detection
            st.subheader("🔍 Data Drift Analysis")
            st.caption("Detects if recent predictions differ significantly from training data distribution")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                min_predictions = st.number_input(
                    "Minimum predictions for analysis:",
                    min_value=5,
                    max_value=100,
                    value=10,
                    step=5
                )
            
            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("🔍 Check for Drift", use_container_width=True, type="primary"):
                    if len(backend.predictions_history) >= min_predictions:
                        with st.spinner("🔄 Analyzing data drift..."):
                            recent_data = backend.predictions_history[-min_predictions:]
                            drift_result = backend.detect_drift(recent_data)
                        
                        st.session_state['drift_result'] = drift_result
                    else:
                        st.warning(f"⚠️ Need at least {min_predictions} predictions. Currently have {len(backend.predictions_history)}.")
            
            with col3:
                st.info("""
                **What is Data Drift?**
                Statistical changes in input features over time that can reduce model accuracy.
                
                **When to check:** After 20+ predictions or weekly
                """)
            
            # Display drift results
            if 'drift_result' in st.session_state:
                drift_result = st.session_state['drift_result']
                
                st.divider()
                
                if drift_result['success']:
                    if drift_result['drift_detected']:
                        st.error("⚠️ **Data Drift Detected!**")
                        st.markdown(drift_result['message'])
                        
                        st.warning("""
                        ### 🔧 Recommended Actions:
                        1. Review recent customer data for changes
                        2. Upload feedback data in the sidebar
                        3. Retrain models with updated data
                        4. Continue monitoring after retraining
                        """)
                    else:
                        st.success("✅ **No Data Drift Detected**")
                        st.markdown(drift_result['message'])
                else:
                    st.info(drift_result.get('message', 'Drift analysis completed'))
            
            st.divider()
            
            # Model Usage Statistics
            st.subheader("📊 Model Usage Statistics")
            
            model_usage = {}
            for p in backend.predictions_history:
                model_name = p['model']
                model_usage[model_name] = model_usage.get(model_name, 0) + 1
            
            if model_usage:
                usage_df = pd.DataFrame(list(model_usage.items()), columns=['Model', 'Count'])
                usage_df = usage_df.sort_values('Count', ascending=False)
                
                fig = px.pie(
                    usage_df,
                    values='Count',
                    names='Model',
                    title='Prediction Distribution by Model'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("📊 No predictions yet. Make predictions in the 'Single Prediction' or 'Batch Predictions' tabs to enable monitoring.")
            
            st.divider()
            
            st.markdown("""
            ### 🔔 What You'll See Here:
            
            - **📈 Prediction Statistics**: Total predictions, churn rate, trends
            - **📊 Timeline Analysis**: Visual timeline of all predictions
            - **🔍 Data Drift Detection**: Monitor for distribution changes
            - **📋 Recent Predictions**: Detailed history of predictions
            - **📊 Model Usage**: Which models are being used most
            
            Start making predictions to populate this dashboard!
            """)
    
    with tab7:
        st.header("📄 Comprehensive Reports")
        
        if st.session_state.current_prediction:
            result = st.session_state.current_prediction
            
            # Report Header
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='color: white; margin: 0 0 10px 0;'>📊 Customer Churn Analysis Report</h2>
                <p style='color: white; margin: 5px 0; opacity: 0.95;'><strong>Generated:</strong> {result['generated_at']}</p>
                <p style='color: white; margin: 5px 0; opacity: 0.95;'><strong>Model Used:</strong> {result['prediction']['model_name']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Executive Summary
            st.subheader("📌 Executive Summary")
            
            churn_status = "HIGH RISK - IMMEDIATE ACTION REQUIRED" if result['prediction']['churn_prediction'] == 1 else "LOW RISK - CUSTOMER STABLE"
            status_color = "danger-box" if result['prediction']['churn_prediction'] == 1 else "success-box"
            
            # Better color scheme for status boxes
            if result['prediction']['churn_prediction'] == 1:
                status_bg = "#fff3cd"  # Light yellow
                status_border = "#ffc107"  # Amber
                status_text = "#000000"  # Black text
            else:
                status_bg = "#d1ecf1"  # Light blue
                status_border = "#0dcaf0"  # Cyan
                status_text = "#000000"  # Black text

           st.markdown(f"""
           <div style='background-color: {status_bg}; 
                       border: 2px solid {status_border}; 
                       border-radius: 5px; 
                       padding: 15px; 
                       margin: 10px 0;
                       color: {status_text};'>
               <h3 style='color: {status_text}; margin-top: 0;'>🎯 Prediction: {churn_status}</h3>
               <p style='color: {status_text};'><strong>Churn Probability:</strong> {result['prediction']['churn_probability']:.1%}</p>
               <p style='color: {status_text};'><strong>Risk Level:</strong> {result['business_impact']['risk_level']}</p>
               <p style='color: {status_text};'><strong>Revenue at Risk:</strong> ${result['business_impact']['revenue_loss']:,.2f}</p>
           </div>
           """, unsafe_allow_html=True)
            
            st.divider()
            
            # Detailed Report
            report_text = f"""
## 📊 Detailed Analysis

### 1. Prediction Results

**Primary Model:** {result['prediction']['model_name']}
- **Churn Probability:** {result['prediction']['churn_probability']:.1%}
- **Binary Prediction:** {'CHURN' if result['prediction']['churn_prediction'] == 1 else 'RETAIN'}
- **Threshold Used:** {result['prediction'].get('threshold_used', 0.5):.2f}

**Best Performing Model:** {result['prediction']['best_model_name']}
- **Churn Probability:** {result['prediction']['best_model_probability']:.1%}
- **Binary Prediction:** {'CHURN' if result['prediction']['best_model_prediction'] == 1 else 'RETAIN'}

**Model Agreement:** {'✅ Models Agree' if result['prediction']['agreement'] else '⚠️ Models Disagree - Review Carefully'}

---

### 2. Business Impact Assessment

**Financial Metrics:**
- **Monthly Charges:** ${result['business_impact']['monthly_charges']:.2f}
- **Customer Lifetime Value:** ${result['business_impact']['lifetime_value']:,.2f}
- **Revenue at Risk:** ${result['business_impact']['revenue_loss']:,.2f}
- **Loss Percentage:** {result['business_impact']['loss_percentage']:.1f}% of total lifetime value
- **Risk Classification:** {result['business_impact']['risk_level']} Risk

**Risk Assessment:**
"""
            
            if result['business_impact']['risk_level'] == 'High':
                report_text += """
🔴 **HIGH RISK** - This customer requires immediate intervention to prevent churn.
Recommended timeline: 24-48 hours for executive-level outreach.
"""
            elif result['business_impact']['risk_level'] == 'Medium':
                report_text += """
🟡 **MEDIUM RISK** - Proactive retention measures strongly recommended.
Recommended timeline: 1-2 weeks for targeted retention campaign.
"""
            else:
                report_text += """
🟢 **LOW RISK** - Customer appears stable. Continue standard relationship nurturing.
Recommended timeline: Routine quarterly check-ins.
"""
            
            report_text += "\n\n---\n\n### 3. Key Risk Factors (SHAP Analysis)\n\n"
            
            if result['shap_explanation']['success']:
                report_text += "**Top Features Contributing to Churn Risk:**\n\n"
                for i, feat in enumerate(result['shap_explanation']['top_features'][:5], 1):
                    direction = "**INCREASES**" if feat['shap_value'] > 0 else "**DECREASES**"
                    impact_icon = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
                    report_text += f"{i}. {impact_icon.get(feat['impact'], '⚪')} **{feat['name']}** ({feat['impact']} Impact)\n"
                    report_text += f"   - SHAP Value: {feat['shap_value']:.4f} ({direction} churn risk)\n"
                    report_text += f"   - Customer's Value: {feat['feature_value']:.2f}\n\n"
            else:
                report_text += "*SHAP analysis unavailable for this prediction.*\n\n"
            
            report_text += "---\n\n### 4. All Model Predictions\n\n"
            
            if result['prediction'].get('all_model_predictions'):
                all_preds = result['prediction']['all_model_predictions']
                report_text += "| Model | Probability | Prediction | Threshold |\n"
                report_text += "|-------|-------------|------------|----------|\n"
                for model_name, pred in all_preds.items():
                    pred_text = 'CHURN' if pred['prediction'] == 1 else 'RETAIN'
                    report_text += f"| {model_name} | {pred['probability']:.1%} | {pred_text} | {pred.get('threshold', 0.5):.2f} |\n"
                
                consensus_rate = sum(p['prediction'] for p in all_preds.values()) / len(all_preds)
                report_text += f"\n**Consensus Rate:** {consensus_rate:.0%} of models predict churn\n"
            
            report_text += "\n---\n\n### 5. Model Performance Metrics\n\n"
            
            if backend.model_metrics.get(result['prediction']['model_name']):
                metrics = backend.model_metrics[result['prediction']['model_name']]
                report_text += f"""
**{result['prediction']['model_name']} Performance:**
- **Accuracy:** {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)
- **Precision:** {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)
- **Recall:** {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)
- **F1-Score:** {metrics['f1']:.3f} ({metrics['f1']*100:.1f}%)
- **ROC-AUC:** {metrics['roc_auc']:.3f}
- **Training Samples:** {metrics.get('training_samples', 'N/A')}
"""
            
            st.markdown(report_text)
            
            st.divider()
            
            # AI Insights Section
            if result['llm_insights']['success']:
                st.subheader("🤖 AI-Generated Business Insights & Recommendations")
                
                if result['llm_insights'].get('fallback'):
                    st.info("💡 Rule-based insights (Configure Gemini API for AI-powered analysis)")
                else:
                    st.success("✨ AI-powered insights from Gemini")
                
                st.markdown(result['llm_insights']['insights'])
            
            st.divider()
            
            # Export Section
            st.subheader("📥 Export Options")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Full JSON Report
                json_str = json.dumps(result, indent=2, default=str)
                st.download_button(
                    "📄 Download Full Report (JSON)",
                    json_str,
                    file_name=f"churn_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # Text Report
                text_report = f"""
CUSTOMER CHURN ANALYSIS REPORT
{'='*80}
Generated: {result['generated_at']}
Model: {result['prediction']['model_name']}

PREDICTION RESULTS
{'-'*80}
Churn Probability: {result['prediction']['churn_probability']:.1%}
Binary Prediction: {'CHURN' if result['prediction']['churn_prediction'] == 1 else 'RETAIN'}
Risk Level: {result['business_impact']['risk_level']}

BUSINESS IMPACT
{'-'*80}
Monthly Charges: ${result['business_impact']['monthly_charges']:.2f}
Revenue at Risk: ${result['business_impact']['revenue_loss']:,.2f}
Lifetime Value: ${result['business_impact']['lifetime_value']:,.2f}
Loss Percentage: {result['business_impact']['loss_percentage']:.1f}%

TOP RISK FACTORS
{'-'*80}
"""
                if result['shap_explanation']['success']:
                    for i, feat in enumerate(result['shap_explanation']['top_features'][:5], 1):
                        direction = "INCREASES" if feat['shap_value'] > 0 else "DECREASES"
                        text_report += f"{i}. {feat['name']} ({feat['impact']} Impact)\n"
                        text_report += f"   SHAP: {feat['shap_value']:.4f} ({direction} risk)\n"
                        text_report += f"   Value: {feat['feature_value']:.2f}\n\n"
                
                text_report += f"\n{'='*80}\n"
                
                st.download_button(
                    "📝 Download Text Report",
                    text_report,
                    file_name=f"churn_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col3:
                # Model Metrics
                if backend.model_metrics:
                    metrics_json = json.dumps(backend.model_metrics, indent=2, default=str)
                    st.download_button(
                        "📊 Download Model Metrics",
                        metrics_json,
                        file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            with col4:
                # Prediction Summary CSV
                summary_data = {
                    'Metric': [
                        'Churn Probability',
                        'Prediction',
                        'Risk Level',
                        'Monthly Charges',
                        'Revenue at Risk',
                        'Lifetime Value',
                        'Model Used',
                        'Timestamp'
                    ],
                    'Value': [
                        f"{result['prediction']['churn_probability']:.1%}",
                        'CHURN' if result['prediction']['churn_prediction'] == 1 else 'RETAIN',
                        result['business_impact']['risk_level'],
                        f"${result['business_impact']['monthly_charges']:.2f}",
                        f"${result['business_impact']['revenue_loss']:,.2f}",
                        f"${result['business_impact']['lifetime_value']:,.2f}",
                        result['prediction']['model_name'],
                        result['generated_at']
                    ]
                }
                summary_csv = pd.DataFrame(summary_data).to_csv(index=False)
                st.download_button(
                    "📋 Download Summary (CSV)",
                    summary_csv,
                    file_name=f"prediction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.divider()
            
            # Additional Analytics
            st.subheader("📊 Additional Analytics")
            
            with st.expander("📈 View Detailed Confusion Matrix"):
                if backend.model_metrics.get(result['prediction']['model_name']):
                    cm = backend.model_metrics[result['prediction']['model_name']].get('confusion_matrix')
                    if cm:
                        st.write("**Confusion Matrix:**")
                        cm_df = pd.DataFrame(
                            cm,
                            columns=['Predicted Retain', 'Predicted Churn'],
                            index=['Actual Retain', 'Actual Churn']
                        )
                        st.dataframe(cm_df, use_container_width=True)
                        
                        # Visualize confusion matrix
                        fig = px.imshow(
                            cm,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=['Retain', 'Churn'],
                            y=['Retain', 'Churn'],
                            title='Confusion Matrix Heatmap',
                            color_continuous_scale='Blues',
                            text_auto=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 View Classification Report"):
                if backend.model_metrics.get(result['prediction']['model_name']):
                    class_report = backend.model_metrics[result['prediction']['model_name']].get('classification_report')
                    if class_report:
                        st.text(class_report)
        
        else:
            st.info("📊 Make a prediction first to generate a comprehensive report")
            
            st.divider()
            
            st.markdown("""
            ### 📄 What's Included in Reports:
            
            #### 📌 Executive Summary
            - Risk level assessment
            - Key metrics at a glance
            - Immediate action recommendations
            
            #### 📊 Detailed Analysis
            - Complete prediction results
            - All model comparisons
            - Model agreement analysis
            
            #### 💰 Business Impact
            - Revenue calculations
            - Customer lifetime value
            - ROI analysis for retention
            
            #### 🔍 Feature Analysis
            - SHAP explanations
            - Top risk factors
            - Feature contributions
            
            #### 🤖 AI Insights
            - Retention strategies
            - Root cause analysis
            - Prioritized action items
            
            #### 📥 Multiple Export Formats
            - JSON (Full data)
            - Text (Readable report)
            - CSV (Summary data)
            - Model metrics
            
            Start making predictions to generate comprehensive reports!
            """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🎯 Customer Churn Prediction Platform v2.0</p>
    <p>Built with Streamlit | Powered by Machine Learning & AI</p>
    <p>© 2026 | Production-Ready Enterprise Solution</p>
</div>
""", unsafe_allow_html=True)
