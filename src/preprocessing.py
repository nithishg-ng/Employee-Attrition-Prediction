import pandas as pd

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

    df = df.fillna(preprocessing_objects["fill_values"])

 