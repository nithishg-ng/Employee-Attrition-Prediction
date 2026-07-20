import pandas as pd

from app.schemas import EmployeeFeatures


def add_engineered_features(d: pd.DataFrame) -> pd.DataFrame:
    
    d = d.copy()
    d["TenureRatio"] = d["YearsAtCompany"] / (d["TotalWorkingYears"] + 1)
    d["PromotionGap"] = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1)
    d["IncomePerYear"] = d["MonthlyIncome"] / (d["TotalWorkingYears"] + 1)
    d["IsOverworked"] = (d["avg_work_hours"] > 8).astype(int)
    d["ManagerStability"] = d["YearsWithCurrManager"] / (d["YearsAtCompany"] + 1)
    return d


def transform(employee: EmployeeFeatures, preprocessing_objects: dict) -> pd.DataFrame:
    
    df = pd.DataFrame([employee.model_dump()])

    # 1. Missing value imputation (values learned from TRAIN in the notebook)
    fill_values = preprocessing_objects["fill_values"]
    df = df.fillna(fill_values)
    # A single-row frame with a None value infers an `object` dtype column even
    # after fillna; force back to numeric so downstream sklearn/XGBoost accept it.
    df[list(fill_values.keys())] = df[list(fill_values.keys())].astype(float)

    # 2. Feature engineering
    df = add_engineered_features(df)

    # 3. Categorical encoding
    cat_cols = preprocessing_objects["cat_cols"]
    encoders = preprocessing_objects["encoder"]

    freq_maps = encoders.get("__freq_maps__", {})
    for col, freq in freq_maps.items():
        df[col] = df[col].map(freq).fillna(0)

    binary_cols = [c for c in cat_cols if c in encoders and c != "__freq_maps__"]
    for col in binary_cols:
        le = encoders[col]
        df[col] = le.transform(df[col])

    onehot_cols = [c for c in cat_cols if c not in binary_cols and c not in freq_maps]
    if onehot_cols:
        df = pd.get_dummies(df, columns=onehot_cols, drop_first=False)
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

    # 4. Align to the exact training column set/order (missing dummy cols -> 0)
    feature_order = preprocessing_objects["feature_order"]
    df = df.reindex(columns=feature_order, fill_value=0)

    # 5. Scaling (fit on TRAIN, applied here as transform-only)
    scaler = preprocessing_objects["scaler"]
    scale_cols = preprocessing_objects["scale_cols"]
    df[scale_cols] = scaler.transform(df[scale_cols])

    return df
