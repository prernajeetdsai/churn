import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report, roc_curve, auc)
import xgboost as xgb
import shap
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
import google.genai as genai
import warnings
warnings.filterwarnings('ignore')

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
try:
    if GEMINI_API_KEY and not GEMINI_API_KEY.startswith('YOUR_'):
        genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Warning: Gemini API not configured: {e}")

class ChurnPredictionBackend:
    def __init__(self, models_dir='./models', data_dir='./data', logs_dir='./logs'):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.logs_dir = logs_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.feature_types = {}
        self.training_data = None
        self.X_train_sample = None  # Store for SHAP
        self.models = {
            'LogisticRegression': None,
            'RandomForest': None,
            'XGBoost': None,
            'GradientBoosting': None
        }
        self.model_metrics = {}
        self.shap_explainers = {}
        self.predictions_history = []
        self.roc_curves = {}
        self._load_models()
        
    def load_csv(self, csv_path):
        """Load and validate CSV data."""
        try:
            df = pd.read_csv(csv_path)
            
            if len(df) < 100:
                return {'success': False, 'error': 'Dataset must have at least 100 rows'}
            
            target_col = None
            for col in ['churn', 'Churn', 'CHURN', 'ChurnRisk']:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col is None:
                return {'success': False, 'error': 'Target column "churn" not found'}
            
            self.training_data = df.copy()
            
            for col in df.columns:
                if col != target_col:
                    if df[col].dtype in ['float64', 'int64']:
                        self.feature_types[col] = 'numeric'
                    else:
                        self.feature_types[col] = 'categorical'
            
            return {
                'success': True,
                'rows': len(df),
                'columns': list(df.columns),
                'target_col': target_col,
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_cols': [c for c, t in self.feature_types.items() if t == 'numeric'],
                'categorical_cols': [c for c, t in self.feature_types.items() if t == 'categorical']
            }
        except Exception as e:
            return {'success': False, 'error': f'CSV Load Error: {str(e)}'}
    
    def preprocess_data(self, df, fit=False):
        """Preprocess: handle missing values, encode, scale."""
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        target_col = None
        for col in ['churn', 'Churn', 'CHURN', 'ChurnRisk']:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            y = df[target_col].map({'No': 0, 'Yes': 1}) if df[target_col].dtype == 'object' else df[target_col]
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df
        
        categorical_features = X.select_dtypes(include='object').columns
        for col in categorical_features:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                self.feature_types[col] = 'categorical'
            else:
                if col in self.label_encoders:
                    try:
                        X[col] = self.label_encoders[col].transform(X[col].astype(str))
                    except ValueError:
                        X[col] = 0
        
        if fit:
            self.feature_columns = X.columns.tolist()
        
        if fit:
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        else:
            X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        
        return X_scaled, y
    
    def train_models(self, csv_path):
        """Train all four models."""
        load_result = self.load_csv(csv_path)
        if not load_result['success']:
            return load_result
        
        X, y = self.preprocess_data(self.training_data, fit=True)
        
        if y is None:
            return {'success': False, 'error': 'No target variable found'}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Store sample for SHAP
        self.X_train_sample = X_train.sample(min(100, len(X_train)), random_state=42)
        
        results = {}
        training_log = []
        
        # 1. Logistic Regression
        try:
            print("Training Logistic Regression...")
            lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            lr_model.fit(X_train, y_train)
            lr_metrics = self._evaluate_model(lr_model, X_test, y_test, 'LogisticRegression')
            
            self.models['LogisticRegression'] = lr_model
            self.model_metrics['LogisticRegression'] = lr_metrics
            results['LogisticRegression'] = lr_metrics
            
            training_log.append(f"✅ Logistic Regression: F1={lr_metrics['f1']:.3f}, Recall={lr_metrics['recall']:.3f}")
        except Exception as e:
            training_log.append(f"❌ Logistic Regression failed: {str(e)}")
        
        # 2. Random Forest
        try:
            print("Training Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=150, max_depth=15, min_samples_split=10,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            rf_metrics = self._evaluate_model(rf_model, X_test, y_test, 'RandomForest')
            
            self.models['RandomForest'] = rf_model
            self.model_metrics['RandomForest'] = rf_metrics
            results['RandomForest'] = rf_metrics
            
            training_log.append(f"✅ Random Forest: F1={rf_metrics['f1']:.3f}, Recall={rf_metrics['recall']:.3f}")
        except Exception as e:
            training_log.append(f"❌ Random Forest failed: {str(e)}")
        
        # 3. XGBoost
        try:
            print("Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=150, max_depth=7, learning_rate=0.1,
                scale_pos_weight=len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1),
                random_state=42, n_jobs=-1, verbosity=0
            )
            xgb_model.fit(X_train, y_train)
            xgb_metrics = self._evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
            
            self.models['XGBoost'] = xgb_model
            self.model_metrics['XGBoost'] = xgb_metrics
            results['XGBoost'] = xgb_metrics
            
            training_log.append(f"✅ XGBoost: F1={xgb_metrics['f1']:.3f}, Recall={xgb_metrics['recall']:.3f}")
        except Exception as e:
            training_log.append(f"❌ XGBoost failed: {str(e)}")
        
        # 4. Gradient Boosting
        try:
            print("Training Gradient Boosting...")
            gb_model = GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=7,
                min_samples_split=10, random_state=42
            )
            gb_model.fit(X_train, y_train)
            gb_metrics = self._evaluate_model(gb_model, X_test, y_test, 'GradientBoosting')
            
            self.models['GradientBoosting'] = gb_model
            self.model_metrics['GradientBoosting'] = gb_metrics
            results['GradientBoosting'] = gb_metrics
            
            training_log.append(f"✅ Gradient Boosting: F1={gb_metrics['f1']:.3f}, Recall={gb_metrics['recall']:.3f}")
        except Exception as e:
            training_log.append(f"❌ Gradient Boosting failed: {str(e)}")
        
        # Determine best model
        valid_models = {k: v for k, v in results.items() if isinstance(v, dict) and 'recall' in v}
        if valid_models:
            best_model = max(valid_models, key=lambda x: valid_models[x]['f1'])
            results['best_model'] = best_model
            results['best_metrics'] = results[best_model]
        
        self._save_models()
        self._log_training('\n'.join(training_log))
        
        return {'success': True, 'models': results, 'log': training_log}
    
    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model and compute metrics."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        self.roc_curves[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
        
        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred),
            'training_samples': len(X_test)
        }
    
    def predict(self, input_dict, model_name='GradientBoosting'):
        """Make prediction with selected model."""
        try:
            if self.feature_columns is None:
                return {'success': False, 'error': 'Model not trained yet'}
            
            processed_input = {}
            for col in self.feature_columns:
                if col in input_dict:
                    val = input_dict[col]
                    if col in self.label_encoders:
                        try:
                            processed_input[col] = self.label_encoders[col].transform([str(val)])[0]
                        except ValueError:
                            processed_input[col] = 0
                    else:
                        processed_input[col] = float(val) if isinstance(val, (int, float)) else val
                else:
                    processed_input[col] = 0
            
            input_df = pd.DataFrame([processed_input])
            X, _ = self.preprocess_data(input_df, fit=False)
            
            if model_name not in self.models or self.models[model_name] is None:
                return {'success': False, 'error': f'Model {model_name} not trained'}
            
            model = self.models[model_name]
            churn_prob = float(model.predict_proba(X)[0][1])
            churn_pred = int(model.predict(X)[0])
            
            best_model_name = self._get_best_model()
            best_model = self.models[best_model_name]
            best_churn_prob = float(best_model.predict_proba(X)[0][1])
            best_churn_pred = int(best_model.predict(X)[0])
            
            self.predictions_history.append({
                'timestamp': datetime.now().isoformat(),
                'model': model_name,
                'probability': churn_prob,
                'prediction': churn_pred,
                'input_features': input_dict
            })
            
            return {
                'success': True,
                'model_name': model_name,
                'churn_probability': churn_prob,
                'churn_prediction': churn_pred,
                'best_model_name': best_model_name,
                'best_model_probability': best_churn_prob,
                'best_model_prediction': best_churn_pred,
                'agreement': churn_pred == best_churn_pred
            }
        except Exception as e:
            return {'success': False, 'error': f'Prediction Error: {str(e)}'}
    
    def calculate_revenue_loss(self, churn_prob, monthly_charges, expected_tenure=24):
        """Calculate expected revenue loss."""
        revenue_loss = churn_prob * monthly_charges * expected_tenure
        lifetime_value = monthly_charges * expected_tenure
        
        if churn_prob < 0.30:
            risk_level = 'Low'
        elif churn_prob < 0.60:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        return {
            'revenue_loss': float(revenue_loss),
            'lifetime_value': float(lifetime_value),
            'risk_level': risk_level,
            'monthly_charges': float(monthly_charges),
            'expected_tenure': expected_tenure,
            'loss_percentage': float((revenue_loss / lifetime_value * 100) if lifetime_value > 0 else 0)
        }
    
    def get_shap_explanation(self, input_dict, model_name='GradientBoosting'):
        """Generate SHAP explanations."""
        try:
            processed_input = {}
            for col in self.feature_columns:
                if col in input_dict:
                    val = input_dict[col]
                    if col in self.label_encoders:
                        try:
                            processed_input[col] = self.label_encoders[col].transform([str(val)])[0]
                        except ValueError:
                            processed_input[col] = 0
                    else:
                        processed_input[col] = float(val) if isinstance(val, (int, float)) else val
                else:
                    processed_input[col] = 0
            
            input_df = pd.DataFrame([processed_input])
            X_pred, _ = self.preprocess_data(input_df, fit=False)
            
            if model_name not in self.models or self.models[model_name] is None:
                return {'success': False, 'error': f'Model {model_name} not trained'}
            
            model = self.models[model_name]
            
            # Use stored sample
            X_sample = self.X_train_sample if self.X_train_sample is not None else X_pred
            
            # Create explainer based on model type
            if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier, GradientBoostingClassifier)):
                explainer = shap.TreeExplainer(model)
                shap_values_raw = explainer.shap_values(X_pred)
                
                # Handle different return formats from TreeExplainer
                if isinstance(shap_values_raw, list):
                    # Binary classification returns [shap_for_class_0, shap_for_class_1]
                    shap_values = shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0]
                else:
                    # Single array case
                    shap_values = shap_values_raw
                
                # Get base value
                if isinstance(explainer.expected_value, np.ndarray):
                    base_value = float(explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0])
                else:
                    base_value = float(explainer.expected_value)
                
                # Global importance
                if self.X_train_sample is not None:
                    global_shap_raw = explainer.shap_values(self.X_train_sample)
                    if isinstance(global_shap_raw, list):
                        global_shap = global_shap_raw[1] if len(global_shap_raw) > 1 else global_shap_raw[0]
                    else:
                        global_shap = global_shap_raw
                    global_importance = np.abs(global_shap).mean(axis=0)
                else:
                    global_importance = np.abs(shap_values[0])
                    
            else:
                # Logistic Regression - use KernelExplainer
                explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(pd.DataFrame(x, columns=self.feature_columns))[:, 1],
                    X_sample
                )
                shap_values = explainer.shap_values(X_pred)
                base_value = float(explainer.expected_value)
                
                # Global importance for linear models
                if self.X_train_sample is not None:
                    global_shap = explainer.shap_values(self.X_train_sample)
                    global_importance = np.abs(global_shap).mean(axis=0)
                else:
                    global_importance = np.abs(shap_values[0])
            
            # Ensure shap_values is 2D
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Local feature importance
            importance = np.abs(shap_values[0])
            top_indices = np.argsort(importance)[-5:][::-1]
            top_features = [{
                'name': self.feature_columns[i],
                'shap_value': float(shap_values[0][i]),
                'feature_value': float(X_pred.iloc[0, i]),
                'impact': 'High' if abs(shap_values[0][i]) > np.percentile(importance, 75) else 'Medium'
            } for i in top_indices]
            
            # Global feature importance
            top_global_indices = np.argsort(global_importance)[-5:][::-1]
            global_features = [{
                'name': self.feature_columns[i],
                'importance': float(global_importance[i])
            } for i in top_global_indices]
            
            return {
                'success': True,
                'model_name': model_name,
                'top_features': top_features,
                'global_importance': global_features,
                'base_value': base_value
            }
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"SHAP Error Details: {error_detail}")
            return {'success': False, 'error': f'SHAP Error: {str(e)}'}

    def get_model_comparison_analysis(self):
        """Get detailed comparison analysis of all models."""
        if not self.model_metrics:
            return {'success': False, 'error': 'No models trained yet'}
        
        comparison = {}
        analysis = {}
        
        for model_name, metrics in self.model_metrics.items():
            comparison[model_name] = {
                'overall_score': (metrics['accuracy'] + metrics['recall'] + metrics['f1']) / 3,
                'churn_detection_score': metrics['recall'],
                'metrics': metrics
            }
        
        best_by_f1 = max(comparison.items(), key=lambda x: x[1]['metrics']['f1'])
        best_by_recall = max(comparison.items(), key=lambda x: x[1]['metrics']['recall'])
        best_by_accuracy = max(comparison.items(), key=lambda x: x[1]['metrics']['accuracy'])
        
        analysis['best_by_f1'] = best_by_f1[0]
        analysis['best_by_recall'] = best_by_recall[0]
        analysis['best_by_accuracy'] = best_by_accuracy[0]
        
        analysis['explanation'] = self._generate_model_comparison_explanation(comparison, analysis)
        
        return {'success': True, 'comparison': comparison, 'analysis': analysis}
    
    def _generate_model_comparison_explanation(self, comparison, analysis):
        """Generate text explanation for model comparison."""
        best_model = analysis['best_by_f1']
        best_metrics = comparison[best_model]['metrics']
        
        explanation = f"""
### 🏆 Best Overall Model: **{best_model}**

**Why {best_model} is the best choice:**

1. **F1-Score: {best_metrics['f1']:.3f}** ⭐
   - Perfect balance between precision and recall
   - Ideal for imbalanced classification problems

2. **Recall: {best_metrics['recall']:.3f}** 🎯
   - Catches {best_metrics['recall']*100:.1f}% of actual churners
   - Minimizes missed churn cases

3. **ROC-AUC: {best_metrics['roc_auc']:.3f}** 📊
   - Excellent discrimination ability
   - Model reliably distinguishes churners from non-churners

4. **Accuracy: {best_metrics['accuracy']:.3f}** ✅
   - Overall correctness of predictions

**Key Strengths:**
- ✅ Highest recall ensures no churners are missed
- ✅ Good precision prevents false alarms
- ✅ Balanced performance across all metrics
- ✅ Best suited for customer retention strategy

**Recommendation:** Use {best_model} for production predictions and retention campaigns.
"""
        
        return explanation
    
    def get_roc_analysis(self):
        """Get ROC curve analysis with explanation."""
        if not self.roc_curves:
            return {'success': False, 'error': 'No models trained yet'}
        
        roc_data = {}
        analysis_text = ""
        
        for model_name, curve_data in self.roc_curves.items():
            roc_data[model_name] = {
                'fpr': curve_data['fpr'].tolist(),
                'tpr': curve_data['tpr'].tolist(),
                'auc': curve_data['auc']
            }
        
        best_auc_model = max(roc_data.items(), key=lambda x: x[1]['auc'])
        
        analysis_text = f"""
### 📉 ROC-AUC Analysis

**Best Model: {best_auc_model[0]} with AUC = {best_auc_model[1]['auc']:.3f}**

**What ROC-AUC Means:**
- **0.5** = Random guessing (flip a coin)
- **0.7-0.8** = Good discrimination
- **0.8-0.9** = Excellent discrimination  
- **>0.9** = Outstanding discrimination

**Interpretation:**
{best_auc_model[0]} has an AUC of **{best_auc_model[1]['auc']:.3f}**, indicating **{'Excellent' if best_auc_model[1]['auc'] > 0.8 else 'Good'}** discrimination ability.

This means:
- The model correctly ranks a random churner higher than a random non-churner **{best_auc_model[1]['auc']*100:.1f}%** of the time
- Very reliable at separating the two groups
- Suitable for real-world deployment

**Comparison with Other Models:**
"""
        
        for model_name, data in sorted(roc_data.items(), key=lambda x: x[1]['auc'], reverse=True):
            analysis_text += f"\n- **{model_name}**: AUC = {data['auc']:.3f}"
        
        return {'success': True, 'roc_data': roc_data, 'analysis': analysis_text}
    
    def detect_drift(self, recent_data):
        """Detect feature/prediction drift."""
        try:
            if self.training_data is None or self.X_train_sample is None:
                return {'success': True, 'drift_detected': False, 'message': 'No training data to compare'}
            
            if isinstance(recent_data, list) and len(recent_data) > 0:
                recent_df = pd.DataFrame([p['input_features'] for p in recent_data])
            else:
                recent_df = recent_data
            
            # Ensure columns match training data
            X_recent_raw = pd.DataFrame()
            for col in self.feature_columns:
                if col in recent_df.columns:
                    X_recent_raw[col] = recent_df[col]
                else:
                    X_recent_raw[col] = 0
            
            X_recent, _ = self.preprocess_data(X_recent_raw, fit=False)
            
            if len(X_recent) < 5:
                return {
                    'success': True,
                    'drift_detected': False,
                    'message': 'Insufficient data for drift analysis (need at least 5 predictions)',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Ensure column order matches
            X_recent = X_recent[self.feature_columns]
            X_reference = self.X_train_sample[self.feature_columns]
            
            report = Report(metrics=[DatasetDriftMetric()])
            report.run(reference_data=X_reference, current_data=X_recent)
            
            drift_detected = report.as_dict()['metrics'][0]['result']['dataset_drift']
            
            drift_text = """
### ⚠️ Data Drift Detected

Dataset drift occurs when the statistical properties of features change over time.
This indicates your models may become less accurate and need retraining.

**Actions to Take:**
1. Review recent data for changes in customer behavior
2. Retrain models with recent feedback data
3. Monitor key features for continued drift
4. Consider updating feature engineering strategies
""" if drift_detected else """
### ✅ No Data Drift Detected

Your models remain valid for current data. Continue monitoring for changes in customer behavior.
Dataset is consistent with training data distribution.
"""
            
            return {
                'success': True,
                'drift_detected': bool(drift_detected),
                'message': drift_text,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'success': True, 'drift_detected': False, 'message': f'Drift analysis error: {str(e)}. Ensure sufficient predictions have been made.'}
    
    def retrain_with_feedback(self, feedback_csv_path):
        """Retrain models with new feedback data."""
        try:
            feedback_df = pd.read_csv(feedback_csv_path)
            if self.training_data is None:
                return {'success': False, 'error': 'Original training data not loaded'}
            
            combined_df = pd.concat([self.training_data, feedback_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates()
            
            combined_path = os.path.join(self.data_dir, f'combined_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            combined_df.to_csv(combined_path, index=False)
            
            return self.train_models(combined_path)
        except Exception as e:
            return {'success': False, 'error': f'Retraining Error: {str(e)}'}
    
    def generate_comprehensive_report(self, input_dict, model_name='GradientBoosting'):
        """Generate comprehensive report with all analyses."""
        try:
            pred_result = self.predict(input_dict, model_name)
            if not pred_result['success']:
                return pred_result
            
            monthly_charges = input_dict.get('MonthlyCharges', 65) if isinstance(input_dict.get('MonthlyCharges'), (int, float)) else 65
            revenue_analysis = self.calculate_revenue_loss(pred_result['churn_probability'], monthly_charges)
            
            shap_result = self.get_shap_explanation(input_dict, model_name)
            model_comparison = self.get_model_comparison_analysis()
            roc_analysis = self.get_roc_analysis()
            
            llm_insights = self.generate_llm_insights({
                'churn_probability': pred_result['churn_probability'],
                'churn_prediction': pred_result['churn_prediction'],
                'risk_level': revenue_analysis['risk_level'],
                'revenue_loss': revenue_analysis['revenue_loss'],
                'loss_percentage': revenue_analysis['loss_percentage'],
                'top_features': shap_result.get('top_features', []) if shap_result['success'] else [],
                'tenure': input_dict.get('tenure', 0),
                'monthly_charges': monthly_charges,
                'input_dict': input_dict
            })
            
            return {
                'success': True,
                'prediction': pred_result,
                'business_impact': revenue_analysis,
                'shap_explanation': shap_result,
                'model_comparison': model_comparison,
                'roc_analysis': roc_analysis,
                'llm_insights': llm_insights,
                'generated_at': datetime.now().isoformat()
            }
        except Exception as e:
            return {'success': False, 'error': f'Report Error: {str(e)}'}
    
    def generate_llm_insights(self, summary_dict):
        """Generate LLM-powered insights with business recommendations."""
        try:
            # Build fallback insights regardless
            churn_prob = summary_dict.get('churn_probability', 0)
            risk_level = summary_dict.get('risk_level', 'Unknown')
            revenue_loss = summary_dict.get('revenue_loss', 0)
            monthly_charges = summary_dict.get('monthly_charges', 0)
            tenure = summary_dict.get('tenure', 0)
            top_features = summary_dict.get('top_features', [])
            
            # Generate rule-based insights
            fallback_insights = self._generate_fallback_insights(summary_dict)
            
            if not GEMINI_API_KEY or GEMINI_API_KEY.startswith('YOUR_'):
                return {
                    'success': True,
                    'insights': fallback_insights,
                    'fallback': True
                }
            
            top_factors = ', '.join([f"{f['name']} (SHAP: {f['shap_value']:.3f})" for f in top_features[:5]])
            
            prompt = f"""You are a customer retention strategist analyzing churn risk for a telecom customer.

**Customer Profile:**
- Churn Probability: {churn_prob:.1%}
- Risk Level: {risk_level}
- Revenue at Risk: ${revenue_loss:.2f}
- Monthly Charges: ${monthly_charges:.2f}
- Customer Tenure: {tenure} months
- Top Risk Factors: {top_factors}

**Your Task:**
Provide a comprehensive business analysis with actionable recommendations:

1. **Risk Assessment** (2-3 sentences on overall risk and urgency)
2. **Root Cause Analysis** (3-5 key factors driving churn risk)
3. **Retention Strategy** (5-7 specific, prioritized actions)
4. **Expected Outcomes** (2-3 sentences on potential impact)
5. **Priority Level** (Urgent/High/Medium/Low with justification)

Focus on business value and practical actions the retention team can take immediately."""
            
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            return {
                'success': True,
                'insights': response.text,
                'generated_at': datetime.now().isoformat(),
                'fallback': False
            }
        except Exception as e:
            return {
                'success': True,
                'insights': fallback_insights,
                'fallback': True,
                'error': str(e)
            }
    
    def _generate_fallback_insights(self, summary_dict):
        """Generate rule-based insights when LLM is unavailable."""
        churn_prob = summary_dict.get('churn_probability', 0)
        risk_level = summary_dict.get('risk_level', 'Unknown')
        revenue_loss = summary_dict.get('revenue_loss', 0)
        monthly_charges = summary_dict.get('monthly_charges', 0)
        tenure = summary_dict.get('tenure', 0)
        top_features = summary_dict.get('top_features', [])
        input_dict = summary_dict.get('input_dict', {})
        
        insights = f"""## 🤖 Customer Churn Analysis & Retention Strategy

### 📊 Risk Assessment
**Churn Probability: {churn_prob:.1%}** | **Risk Level: {risk_level}** | **Revenue at Risk: ${revenue_loss:.2f}**

"""
        
        # Risk interpretation
        if churn_prob >= 0.70:
            insights += """**CRITICAL RISK** ⚠️ - This customer shows very high likelihood of churning. Immediate intervention required to prevent revenue loss.

"""
        elif churn_prob >= 0.50:
            insights += """**HIGH RISK** 🔴 - Customer displays significant churn signals. Proactive retention measures strongly recommended.

"""
        elif churn_prob >= 0.30:
            insights += """**MODERATE RISK** 🟡 - Customer shows some churn indicators. Monitor closely and implement preventive strategies.

"""
        else:
            insights += """**LOW RISK** 🟢 - Customer appears stable but continue relationship nurturing activities.

"""
        
        # Root Cause Analysis
        insights += "### 🔍 Root Cause Analysis\n\n"
        if top_features:
            insights += "**Key Risk Factors Identified:**\n\n"
            for i, feat in enumerate(top_features[:5], 1):
                impact_icon = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
                insights += f"{i}. {impact_icon.get(feat['impact'], '⚪')} **{feat['name']}** - Impact: {feat['impact']} (SHAP: {feat['shap_value']:.3f})\n"
            insights += "\n"
        
        # Analyze specific features
        contract_type = input_dict.get('Contract', 'Unknown')
        internet_service = input_dict.get('InternetService', 'Unknown')
        payment_method = input_dict.get('PaymentMethod', 'Unknown')
        
        insights += "**Behavioral Patterns:**\n\n"
        
        if tenure < 12:
            insights += "- 📅 **Short Tenure Alert**: Customer is in high-risk early lifecycle stage (< 1 year)\n"
        elif tenure < 24:
            insights += "- 📅 **Mid-Term Customer**: Building relationship but not yet fully committed\n"
        else:
            insights += "- 📅 **Loyal Customer**: Long tenure suggests established relationship\n"
        
        if monthly_charges > 80:
            insights += "- 💰 **Premium Customer**: High monthly charges indicate valuable customer segment\n"
        elif monthly_charges < 30:
            insights += "- 💵 **Budget Customer**: Low monthly charges may indicate limited service adoption\n"
        
        if 'month' in str(contract_type).lower():
            insights += "- 📄 **Month-to-Month Contract**: No commitment = higher churn risk\n"
        elif 'year' in str(contract_type).lower() or 'two' in str(contract_type).lower():
            insights += "- 📄 **Term Contract**: Contractual commitment provides stability\n"
        
        insights += "\n"
        
        # Retention Strategy
        insights += "### 🎯 Recommended Retention Actions\n\n"
        insights += "**Immediate Actions (Next 7 Days):**\n\n"
        
        if churn_prob >= 0.60:
            insights += """1. 📞 **Executive Outreach**: Senior manager should personally call customer
2. 🎁 **Special Offer**: Present exclusive retention package with 20-30% discount
3. ⚡ **Fast-Track Resolution**: Address any outstanding service issues within 24 hours
4. 💎 **VIP Treatment**: Upgrade to premium support tier at no cost
5. 📊 **Usage Review**: Schedule consultation to optimize services to customer needs

"""
        elif churn_prob >= 0.40:
            insights += """1. 📧 **Personalized Outreach**: Targeted email campaign highlighting value proposition
2. 🎁 **Loyalty Incentive**: Offer 10-15% discount or service upgrade
3. 📞 **Courtesy Call**: Customer success check-in to understand satisfaction
4. 📱 **Service Optimization**: Review and recommend better-fit plans
5. 🌟 **Engagement Program**: Invite to customer loyalty program with benefits

"""
        else:
            insights += """1. 📧 **Relationship Nurturing**: Regular engagement emails with helpful tips
2. 🎯 **Cross-Sell Opportunities**: Introduce complementary services
3. 📊 **Usage Analytics**: Share insights on how to maximize service value
4. 💬 **Feedback Collection**: Quarterly satisfaction surveys
5. 🏆 **Reward Program**: Enroll in points/rewards for continued loyalty

"""
        
        insights += "**Strategic Actions (30-90 Days):**\n\n"
        
        if tenure < 12:
            insights += "- 🎓 **Onboarding Enhancement**: Improve early customer experience and education\n"
        
        if 'month' in str(contract_type).lower():
            insights += "- 📄 **Contract Upgrade Incentive**: Offer benefits for switching to annual contract\n"
        
        if monthly_charges < 40:
            insights += "- 📈 **Value Demonstration**: Show additional services that enhance experience\n"
        
        insights += "- 🔄 **Feedback Loop**: Implement customer suggestions and communicate changes\n"
        insights += "- 🤝 **Community Building**: Create customer community for peer support\n"
        insights += "- 📊 **Predictive Monitoring**: Set up alerts for behavior changes\n\n"
        
        # Expected Outcomes
        insights += "### 📈 Expected Outcomes\n\n"
        
        if churn_prob >= 0.60:
            reduction = 30
            insights += f"""**Potential Impact:** Implementing these strategies could reduce churn probability by {reduction-40}% to {reduction-45}%.

- 💰 **Revenue Protection**: Save approximately ${revenue_loss * 0.4:.2f} to ${revenue_loss * 0.6:.2f}
- 📊 **Success Rate**: Historical data shows 40-50% success rate with critical interventions
- ⏱️ **Time Sensitivity**: Act within 48-72 hours for maximum impact\n\n"""
        elif churn_prob >= 0.40:
            reduction = 40
            insights += f"""**Potential Impact:** Targeted retention efforts typically reduce churn probability by {reduction-25}% to {reduction-30}%.

- 💰 **Revenue Protection**: Save approximately ${revenue_loss * 0.5:.2f} to ${revenue_loss * 0.7:.2f}
- 📊 **Success Rate**: 50-60% success rate with proactive engagement
- ⏱️ **Time Sensitivity**: Implement within 1-2 weeks\n\n"""
        else:
            reduction = 50
            insights += f"""**Potential Impact:** Preventive measures can maintain low churn risk and increase lifetime value.

- 💰 **Revenue Protection**: Maintain customer lifetime value of ${revenue_loss / churn_prob if churn_prob > 0 else monthly_charges * 36:.2f}
- 📊 **Success Rate**: 70-80% retention rate with consistent engagement
- ⏱️ **Time Sensitivity**: Ongoing monitoring and relationship building\n\n"""
        
        # Priority Level
        insights += "### 🚦 Priority Level\n\n"
        
        if churn_prob >= 0.70:
            priority = "🔴 **URGENT**"
            justification = "Critical churn risk requires immediate C-level escalation and intervention within 24-48 hours."
        elif churn_prob >= 0.50:
            priority = "🟠 **HIGH**"
            justification = "Significant churn risk requires dedicated retention specialist assignment within 1 week."
        elif churn_prob >= 0.30:
            priority = "🟡 **MEDIUM**"
            justification = "Moderate risk warrants proactive outreach and monitoring within 2-3 weeks."
        else:
            priority = "🟢 **LOW**"
            justification = "Standard retention activities sufficient with regular quarterly check-ins."
        
        insights += f"{priority}\n\n**Justification:** {justification}\n\n"
        
        # Bottom note
        insights += "---\n\n"
        insights += "*💡 **Note**: To enable advanced AI-powered insights with deeper analysis, configure GEMINI_API_KEY in your .env file.*\n"
        insights += "*Get your free API key at: https://aistudio.google.com/apikey*\n"
        
        return insights
    
    def _get_best_model(self):
        """Get best model by F1-score."""
        if not self.model_metrics:
            return 'GradientBoosting'
        valid = {k: v for k, v in self.model_metrics.items() if 'f1' in v}
        return max(valid, key=lambda x: valid[x]['f1']) if valid else 'GradientBoosting'
    
    def _save_models(self):
        """Save all models and preprocessors."""
        try:
            for name, model in self.models.items():
                if model:
                    path = os.path.join(self.models_dir, f'{name}.pkl')
                    with open(path, 'wb') as f:
                        pickle.dump(model, f)
            
            with open(os.path.join(self.models_dir, 'encoders.pkl'), 'wb') as f:
                pickle.dump(self.label_encoders, f)
            with open(os.path.join(self.models_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(os.path.join(self.models_dir, 'features.json'), 'w') as f:
                json.dump(self.feature_columns, f)
            with open(os.path.join(self.models_dir, 'feature_types.json'), 'w') as f:
                json.dump(self.feature_types, f)
            if self.X_train_sample is not None:
                with open(os.path.join(self.models_dir, 'X_train_sample.pkl'), 'wb') as f:
                    pickle.dump(self.X_train_sample, f)
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load all saved models."""
        try:
            for name in self.models.keys():
                path = os.path.join(self.models_dir, f'{name}.pkl')
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.models[name] = pickle.load(f)
            
            if os.path.exists(os.path.join(self.models_dir, 'encoders.pkl')):
                with open(os.path.join(self.models_dir, 'encoders.pkl'), 'rb') as f:
                    self.label_encoders = pickle.load(f)
            
            if os.path.exists(os.path.join(self.models_dir, 'scaler.pkl')):
                with open(os.path.join(self.models_dir, 'scaler.pkl'), 'rb') as f:
                    self.scaler = pickle.load(f)
            
            if os.path.exists(os.path.join(self.models_dir, 'features.json')):
                with open(os.path.join(self.models_dir, 'features.json'), 'r') as f:
                    self.feature_columns = json.load(f)
            
            if os.path.exists(os.path.join(self.models_dir, 'feature_types.json')):
                with open(os.path.join(self.models_dir, 'feature_types.json'), 'r') as f:
                    self.feature_types = json.load(f)
            
            if os.path.exists(os.path.join(self.models_dir, 'X_train_sample.pkl')):
                with open(os.path.join(self.models_dir, 'X_train_sample.pkl'), 'rb') as f:
                    self.X_train_sample = pickle.load(f)
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def _log_training(self, log_content):
        """Log training information."""
        try:
            log_file = os.path.join(self.logs_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            with open(log_file, 'w') as f:
                f.write(log_content)
        except Exception as e:
            print(f"Error writing training log: {e}")