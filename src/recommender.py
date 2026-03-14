
import pandas as pd
import json

URGENCY_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}

def assign_risk_level(score):
    if score >= 0.70:
        return "High"
    elif score >= 0.40:
        return "Medium"
    else:
        return "Low"

def match_rules(feature_name, feature_value, shap_value, rules):
    matched = []
    for rule in rules:
        if rule["trigger_feature"] != feature_name:
            continue
        condition = rule["condition"]
        threshold = rule["threshold"]
        triggered = False
        if condition == "low"    and feature_value <= threshold: triggered = True
        elif condition == "high" and feature_value >= threshold: triggered = True
        elif condition == "equals" and int(feature_value) == int(threshold): triggered = True
        if triggered:
            matched.append({
                "rule_id"            : rule["id"],
                "risk_category"      : rule["risk_category"],
                "intervention_title" : rule["intervention_title"],
                "intervention_detail": rule["intervention_detail"],
                "responsible_party"  : rule["responsible_party"],
                "urgency"            : rule["urgency"],
                "trigger_feature"    : feature_name,
                "feature_value"      : feature_value,
                "shap_value"         : shap_value
            })
    return matched

def get_recommendations(student_features, student_shap_values, feature_names, rules, top_n_factors=5):
    shap_series = pd.Series(student_shap_values, index=feature_names)
    top_factors = shap_series[shap_series > 0].nlargest(top_n_factors)
    all_recommendations = []
    seen_titles = set()
    for feature_name, shap_val in top_factors.items():
        feature_val = student_features[feature_name]
        matches = match_rules(feature_name, feature_val, shap_val, rules)
        for rec in matches:
            if rec["intervention_title"] not in seen_titles:
                seen_titles.add(rec["intervention_title"])
                rec["priority_score"] = round(shap_val * URGENCY_WEIGHT[rec["urgency"]], 4)
                all_recommendations.append(rec)
    all_recommendations.sort(key=lambda x: x["priority_score"], reverse=True)
    return all_recommendations
