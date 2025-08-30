import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_recall_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt

# ========== Helper Functions ===========

def parse_contig_string(contig_str):
    contigs = []
    segments = contig_str.split('/')
    
    for seg in segments:
        if seg.startswith('A'):
            contig_type = 'fixed_backbone'
            start, end = seg[1:].split('-') 
            length = int(end) - int(start) + 1
        else:
            contig_type = 'designed_loop'
            start, end = seg.split('-')
            length = int(end) - int(start) + 1  

        contigs.append({
            'type': contig_type,
            'start': int(start),
            'end': int(end),
            'length': length
        })
    return contigs

def extract_features_from_contigs(contig_str):
    """Extracts features from a contig string."""
    contigs = parse_contig_string(contig_str)
    
    designed_loops = [c for c in contigs if c['type'] == 'designed_loop']
    fixed_chunks = [c for c in contigs if c['type'] == 'fixed_backbone']
    
    # --- 1. Basic Counts ---
    num_contigs = len(contigs)
    num_designed_loops = len(designed_loops)
    num_fixed_chunks = len(fixed_chunks)
    
    # --- 2. Length Features ---
    total_length = sum(c['length'] for c in contigs)
    total_designed_length = sum(c['length'] for c in designed_loops)
    total_fixed_length = sum(c['length'] for c in fixed_chunks)
    
    # Statistics on loop lengths
    loop_lengths = [c['length'] for c in designed_loops]
    avg_loop_length = np.mean(loop_lengths) if loop_lengths else 0
    min_loop_length = min(loop_lengths) if loop_lengths else 0
    max_loop_length = max(loop_lengths) if loop_lengths else 0
    std_loop_length = np.std(loop_lengths) if loop_lengths else 0
    
    # Statistics on fixed chunk lengths
    fixed_lengths = [c['length'] for c in fixed_chunks]
    avg_fixed_length = np.mean(fixed_lengths) if fixed_lengths else 0
    min_fixed_length = min(fixed_lengths) if fixed_lengths else 0
    max_fixed_length = max(fixed_lengths) if fixed_lengths else 0
    
    # --- 3. Spatial / Positional Features ---
    starts = [c['start'] for c in contigs]
    ends = [c['end'] for c in contigs]
    
    total_span = max(ends) - min(starts) + 1 if contigs else 0
    
    gaps = []
    sorted_contigs = sorted(contigs, key=lambda x: x['start'])
    for i in range(1, len(sorted_contigs)):
        gap = sorted_contigs[i]['start'] - sorted_contigs[i-1]['end'] - 1
        gaps.append(gap)
    
    avg_gap = np.mean(gaps) if gaps else 0
    max_gap = max(gaps) if gaps else 0
    min_gap = min(gaps) if gaps else 0
    total_gap = sum(gaps) if gaps else 0
    
    # --- 4. Ratio Features ---
    ratio_designed_to_total = total_designed_length / total_length if total_length else 0
    ratio_fixed_to_total = total_fixed_length / total_length if total_length else 0
    ratio_span_to_length = total_span / total_length if total_length else 0
    
    # --- 5. Pattern Features ---
    segment_pattern = ''.join(['L' if c['type'] == 'designed_loop' else 'F' for c in sorted_contigs])
    
    features = {
        'num_contigs': num_contigs,
        'num_designed_loops': num_designed_loops,
        'num_fixed_chunks': num_fixed_chunks,
        'total_length': total_length,
        'total_designed_length': total_designed_length,
        'total_fixed_length': total_fixed_length,
        'avg_loop_length': avg_loop_length,
        'min_loop_length': min_loop_length,
        'max_loop_length': max_loop_length,
        'std_loop_length': std_loop_length,
        'avg_fixed_length': avg_fixed_length,
        'min_fixed_length': min_fixed_length if fixed_lengths else 0,  
        'max_fixed_length': max_fixed_length,  
        'total_span': total_span,
        'avg_gap': avg_gap,
        'max_gap': max_gap,
        'min_gap': min_gap,
        'total_gap': total_gap,
        'ratio_designed_to_total': ratio_designed_to_total,
        'ratio_fixed_to_total': ratio_fixed_to_total,
        'ratio_span_to_length': ratio_span_to_length,
        'segment_pattern': segment_pattern
    }
    # lines of tab data x 22
    return features

def is_ood(features_dict, X_train_stats):
    """Check if a new sample is Out-of-Distribution."""
    for feature, value in features_dict.items():
        if feature in ['segment_pattern']: 
            continue
        
        # Check if the feature exists in training stats
        if feature not in X_train_stats.columns:
            continue
            
        train_min = X_train_stats.at['min', feature]  
        train_max = X_train_stats.at['max', feature]  

        if value < train_min or value > train_max:
            print(f"OOD Alert: {feature} = {value} is outside training range [{train_min}, {train_max}]")
            return True
    return False

def predict_contig_success(contig_string, pipeline_path='contig_prediction_pipeline.joblib', threshold=0.05):
    """Predict success probability for a contig string."""
    try:
        # Load the full pipeline
        full_pipeline = joblib.load(pipeline_path)
        preprocessor = full_pipeline['preprocessor']
        model = full_pipeline['model']
        
        # Extract features
        features_dict = extract_features_from_contigs(contig_string)
        features_df = pd.DataFrame([features_dict])
        
        # Check for OOD
        if is_ood(features_dict, X_train_stats):
            print("Warning: This configuration is highly unusual and the prediction may be unreliable.")
            return -1, "OOD"
        
        # Preprocess and predict
        new_data_processed = preprocessor.transform(features_df)
        probability = model.predict_proba(new_data_processed)[0, 1]
        prediction = 1 if probability >= threshold else 0
        
        return prediction, probability
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return -1, f"Error: {e}"

def enhance_features(X_df):
    """Create additional features based on feature importance analysis"""
    X_enhanced = X_df.copy()
    
    # Gap-to-span ratio (important based on your feature importance)
    X_enhanced['gap_span_ratio'] = X_enhanced['total_gap'] / X_enhanced['total_span']
    
    # Average gap per contig
    X_enhanced['avg_gap_per_contig'] = X_enhanced['total_gap'] / X_enhanced['num_contigs']
    
    # Pattern complexity (count of transitions between L and F)
    X_enhanced['pattern_complexity'] = X_enhanced['segment_pattern'].apply(
        lambda x: sum(1 for i in range(1, len(x)) if x[i] != x[i-1])
    )
    
    # Dominant segment type
    X_enhanced['dominant_type'] = X_enhanced['segment_pattern'].apply(
        lambda x: 'L' if x.count('L') > x.count('F') else 'F'
    )
    
    return X_enhanced

# Find the threshold that gives you the best precision while maintaining reasonable recall
def find_optimal_precision_threshold(precisions, recalls, thresholds, min_recall=0.3):
    """Find threshold that maximizes precision while maintaining minimum recall"""
    valid_indices = (recalls[:-1] >= min_recall)  # Ensure minimum recall
    if np.any(valid_indices):
        best_idx = np.argmax(precisions[:-1][valid_indices])
        best_threshold = thresholds[valid_indices][best_idx]
        best_precision = precisions[:-1][valid_indices][best_idx]
        best_recall = recalls[:-1][valid_indices][best_idx]
        return best_threshold, best_precision, best_recall
    else:
        # Fallback: use threshold with max precision
        best_idx = np.argmax(precisions[:-1])
        return thresholds[best_idx], precisions[:-1][best_idx], recalls[:-1][best_idx]

def pattern_based_heuristic(contig_string):
    """Add pattern-based rules to complement the ML model"""
    features = extract_features_from_contigs(contig_string)
    pattern = features['segment_pattern']
    
    # Successful patterns from your feature importance
    successful_patterns = ['LLLLFFFF', 'LLLLLFFFF', 'LLLLFF', 'LLLLFFF']
    
    if pattern in successful_patterns:
        return 0.7  
    elif pattern.count('L') >= 4:  # Multiple loops often work
        return 0.6
    else:
        return 0.5  # Neutral

# Modify your prediction function:
def predict_contig_success_enhanced(contig_string, pipeline_path='contig_prediction_pipeline_xgb.joblib'):
    prediction, probability = predict_contig_success_xgb(contig_string, pipeline_path)
    
    if prediction != -1:  # If not OOD
        pattern_boost = pattern_based_heuristic(contig_string)
        blended_prob = 0.7 * probability + 0.3 * pattern_boost
        prediction = 1 if blended_prob >= precision_threshold else 0
        return prediction, blended_prob
    
    return prediction, probability

def predict_contig_success_xgb(contig_string, pipeline_path='contig_prediction_pipeline_xgb.joblib', use_precision_threshold=True):
    """Predict success probability for a contig string using XGBoost."""
    try:
        # Load the full pipeline
        full_pipeline = joblib.load(pipeline_path)
        preprocessor = full_pipeline['preprocessor']
        model = full_pipeline['model']
        threshold = full_pipeline['precision_threshold'] if use_precision_threshold else full_pipeline['f1_threshold']
        
        # Extract features
        features_dict = extract_features_from_contigs(contig_string)
        features_df = pd.DataFrame([features_dict])
        
        # Enhance features
        features_enhanced = enhance_features(features_df)
        
        # Check for OOD
        if is_ood(features_dict, X_train_stats):
            print("Warning: This configuration is highly unusual and the prediction may be unreliable.")
            return -1, "OOD"
        
        # Preprocess and predict
        new_data_processed = preprocessor.transform(features_enhanced)
        probability = model.predict_proba(new_data_processed)[0, 1]
        prediction = 1 if probability >= threshold else 0
        
        return prediction, probability
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return -1, f"Error: {e}"


# ========== Main Execution ===========

# Load and process data
df = pd.read_csv("/Users/nathangu/Desktop/Pytorch/HSF/RF Data.csv")

feature_list = []
for contig_str in df['Chosen Contigs']:
    feature_list.append(extract_features_from_contigs(contig_str))
    
X_df = pd.DataFrame(feature_list)

# Create target variable
X_df['is_high_quality'] = (
    (df['PLDDT'] >= 0.8) & 
    (df['TM Align Motif Score'] >= 0.80) & 
    (df['TM Align Protein Score'] < 0.6) 
).astype(int)

print("Number of high-quality designs:", X_df['is_high_quality'].sum())
print("Number of low-quality designs:", len(X_df) - X_df['is_high_quality'].sum())
print("Percentage of high-quality designs: {:.2f}%".format(100 * X_df['is_high_quality'].mean()))

# Preprocessing
y = X_df['is_high_quality']
X_raw = X_df.drop('is_high_quality', axis=1)

# Apply feature enhancement
X_enhanced = enhance_features(X_raw)

# Create training statistics BEFORE preprocessing
X_train_stats = X_enhanced.drop('segment_pattern', axis=1).describe().loc[['min', 'max']].T

categorical_feature = ['segment_pattern', 'dominant_type']  
numerical_features = X_enhanced.columns.drop(categorical_feature)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feature)
    ],
    remainder='passthrough'
)

X_final = preprocessor.fit_transform(X_enhanced)
feature_names_out = preprocessor.get_feature_names_out()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

# Calculate the crucial scale_pos_weight parameter
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Using scale_pos_weight: {scale_pos_weight}")

# Use stratified K-fold for cross-validation
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create synthetic examples of the minority class
smote = SMOTE(random_state=42, k_neighbors=min(3, y_train.sum()-1))

# Applies SMOTE then XGBoost
pipeline = ImbPipeline([
    ('smote', smote),
    ('xgb', xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    ))
])

# Parameter grid for XGBoost
param_grid = {
    'xgb__max_depth': [3, 6],
    'xgb__learning_rate': [0.01, 0.1],
    'xgb__subsample': [0.8, 0.9],
    'xgb__gamma': [0, 0.1],
}

# Grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=stratified_kfold,
    scoring='roc_auc',
    n_jobs=1,
    verbose=1
)

print("Training XGBoost with SMOTE and Cross-Validation...")
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
print("\nFinal Evaluation on Test Set:")
print(classification_report(y_test, y_pred, target_names=['Low Quality', 'High Quality']))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Quality', 'High Quality'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - XGBoost with SMOTE')
plt.show()

# Threshold tuning for imbalanced data
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
print(f"Precisions length: {len(precisions)}, Recalls length: {len(recalls)}, Thresholds length: {len(thresholds)}")

# Find threshold that maximizes F1-score 
f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)  # Added small epsilon to avoid division by zero
best_idx = np.nanargmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Best F1 Threshold: {best_threshold:.4f}, Best F1-Score: {best_f1:.4f}")

# Find threshold that gives at least 80% precision
high_precision_mask = precisions[:-1] >= 0.8  

if np.any(high_precision_mask):
    # Get the first threshold that achieves >= 80% precision
    precision_threshold = thresholds[high_precision_mask][0]
    achieved_precision = precisions[:-1][high_precision_mask][0]
    print(f"Threshold for 80% precision: {precision_threshold:.4f} (Precision: {achieved_precision:.4f})")
else:
    # Use threshold that gives max precision
    max_precision_idx = np.argmax(precisions[:-1])
    precision_threshold = thresholds[max_precision_idx]
    max_precision = precisions[:-1][max_precision_idx]
    print(f"Max precision threshold: {precision_threshold:.4f}, Precision: {max_precision:.4f}")


# precision_threshold, achieved_precision, achieved_recall = find_optimal_precision_threshold(
#     precisions, recalls, thresholds, min_recall=0.3
# )
# print(f"Optimal precision threshold: {precision_threshold:.4f}")
# print(f"Precision: {achieved_precision:.4f}, Recall: {achieved_recall:.4f}")

# # PR-AUC
# pr_auc = auc(recalls, precisions)
# print(f"PR-AUC Score: {pr_auc:.4f}")

# # PR curve
# plt.figure(figsize=(8, 6))
# plt.plot(recalls, precisions, marker='.')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.grid(True)
# plt.show()

# Save pipeline with both thresholds
pipeline_obj = {
    'preprocessor': preprocessor,
    'model': best_model,
    'f1_threshold': best_threshold,
    'precision_threshold': precision_threshold
}
joblib.dump(pipeline_obj, 'contig_prediction_pipeline_xgb.joblib')
print("\nSaved XGBoost pipeline to 'contig_prediction_pipeline_xgb.joblib'")


# Cross-validation on full dataset
print("\nCross-validated performance on full dataset:")
y_pred_proba_cv = cross_val_predict(
    pipeline, X_final, y, cv=stratified_kfold, 
    method='predict_proba', n_jobs=1
)[:, 1]

# Precision threshold for CV evaluation
print("Using precision threshold:")
print(classification_report(y, (y_pred_proba_cv >= precision_threshold).astype(int)))

# Feature importance
feature_importances = best_model.named_steps['xgb'].feature_importances_
feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("Top 10 most important features:")
print(importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Test predictions
print("\n" + "="*50)
print("Testing XGBoost Prediction on New Configurations")
print("="*50)

new_configs = [
    "3-3/A68-84/10-10/A125-190/50-50/A230-238/5-5", 
    "55-55/A69-85/27-27/A129-131/18-18/A161-179/48-48", 
    "65-65/A69-85/27-27/A129-131/21-21/A161-179/63-63/A230-237/29-29", 
    "76-76/A69-85/50-50/A129-131/27-27/A161-179/39-39/A230-237/23-23", 
    "53-53/A69-85/58-58/A129-131/38-38/A161-179/65-65/A230-237/16-16", 
    "80-80/A69-85/43-43/A129-131/29-29/A161-179/44-44/A230-237/18-18"
]

for config in new_configs:
    pred, prob = predict_contig_success_xgb(config, use_precision_threshold=True)
    print(f"Contig: {config}")
    if pred == -1:
        print(f"  Status: {prob}")
    else:
        print(f"  Predicted Success: {pred} (Confidence: {prob:.3f})")
        if pred == 1:
            print("  -> Sending to RFdiffusion!")
        else:
            print("  -> Skipping. Low probability of success.")
    print()