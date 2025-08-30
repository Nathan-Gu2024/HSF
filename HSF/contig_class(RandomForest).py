import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
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

# Create training statistics BEFORE preprocessing
X_train_stats = X_raw.drop('segment_pattern', axis=1).describe().loc[['min', 'max']].T 

categorical_feature = ['segment_pattern']
numerical_features = X_raw.columns.drop(categorical_feature)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feature)
    ],
    remainder='passthrough'
)

X_final = preprocessor.fit_transform(X_raw)
feature_names_out = preprocessor.get_feature_names_out()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

# Train model
print("Performing Grid Search...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2],
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=1, scoring='roc_auc')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
print("\nFinal Evaluation on Test Set:")
print(classification_report(y_test, y_pred))
print("Random Forest ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Save pipeline
pipeline = {
    'preprocessor': preprocessor,
    'model': best_model
}
joblib.dump(pipeline, 'contig_prediction_pipeline.joblib')
print("\nSaved preprocessor and model to 'contig_prediction_pipeline.joblib'")

# Test predictions
print("\n" + "="*50)
print("Testing Prediction on New Configurations")
print("="*50)

# new_configs = [
#     "3-3/A68-84/10-10/A125-190/50-50/A230-238/5-5", 
#     "55-55/A69-85/27-27/A129-131/18-18/A161-179/48-48", 
#     "65-65/A69-85/27-27/A129-131/21-21/A161-179/63-63/A230-237/29-29", 
#     "76-76/A69-85/50-50/A129-131/27-27/A161-179/39-39/A230-237/23-23", 
#     "53-53/A69-85/58-58/A129-131/38-38/A161-179/65-65/A230-237/16-16", 
#     "80-80/A69-85/43-43/A129-131/29-29/A161-179/44-44/A230-237/18-18"
# ]

# Patterns that actually worked in your training data
# Key looking for short loops
test_configs_similar_to_success = [
    "40-40/A69-85/25-25/A129-131/30-30/A161-179/35-35",  
    "20-20/A69-85/15-15/A129-131/25-25/A161-179/30-30",  
]

for config in test_configs_similar_to_success:
    pred, prob = predict_contig_success(config)
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

feature_importances = best_model.feature_importances_
feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("Top 10 most important features:")
print(importance_df.head(10))


# Compare successful vs failed designs
successful_designs = X_df[X_df['is_high_quality'] == 1]
failed_designs = X_df[X_df['is_high_quality'] == 0]

print("Successful designs - feature averages:")
print(successful_designs.mean(numeric_only=True))

print("\nFailed designs - feature averages:")
print(failed_designs.mean(numeric_only=True))


# Patterns of successful designs
successful_patterns = X_df[X_df['is_high_quality'] == 1]['segment_pattern'].value_counts()
print("Patterns of successful designs:")
print(successful_patterns)


# "Promising" failures to test next
failed_designs = X_df[X_df['is_high_quality'] == 0].copy()

# Get model probabilities for all training data
failed_designs['predicted_prob'] = best_model.predict_proba(preprocessor.transform(X_raw[X_df['is_high_quality'] == 0]))[:, 1]

# Failures that the model was most uncertain about
uncertain_failures = failed_designs[(failed_designs['predicted_prob'] > 0.2) & (failed_designs['predicted_prob'] < 0.4)]
print("Most promising failures to investigate:")
print(uncertain_failures[['predicted_prob']].head(10))


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
print("Testing Random Forest Prediction on New Configurations")
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
    pred, prob = predict_contig_success(config)
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