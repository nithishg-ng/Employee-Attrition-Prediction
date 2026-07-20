from typing import Literal, Optional

from pydantic import BaseModel, Field


class EmployeeFeatures(BaseModel):
    
    Age: int = Field(..., ge=18, le=60, description="Employee age in years")
    BusinessTravel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    Department: Literal["Sales", "Research & Development", "Human Resources"]
    DistanceFromHome: int = Field(..., ge=1, le=29, description="Distance from home in km")
    Education: int = Field(..., ge=1, le=5, description="1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor")
    EducationField: Literal[
        "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"
    ]
    Gender: Literal["Male", "Female"]
    JobLevel: int = Field(..., ge=1, le=5)
    JobRole: Literal[
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative",
    ]
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    MonthlyIncome: float = Field(..., ge=10000, le=200000)
    NumCompaniesWorked: Optional[float] = Field(None, ge=0, le=9, description="Missing -> imputed with train median")
    PercentSalaryHike: int = Field(..., ge=11, le=25)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: Optional[float] = Field(None, ge=0, le=40, description="Missing -> imputed with train median")
    TrainingTimesLastYear: int = Field(..., ge=0, le=6)
    YearsAtCompany: int = Field(..., ge=0, le=40)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=15)
    YearsWithCurrManager: int = Field(..., ge=0, le=17)
    EnvironmentSatisfaction: Optional[float] = Field(None, ge=1, le=4, description="Missing -> imputed with train mode")
    JobSatisfaction: Optional[float] = Field(None, ge=1, le=4, description="Missing -> imputed with train mode")
    WorkLifeBalance: Optional[float] = Field(None, ge=1, le=4, description="Missing -> imputed with train mode")
    JobInvolvement: int = Field(..., ge=1, le=4)
    PerformanceRating: int = Field(..., ge=1, le=4)
    avg_work_hours: float = Field(..., ge=0, le=24, description="Average daily working hours derived from in_time/out_time logs")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 41,
                "BusinessTravel": "Travel_Rarely",
                "Department": "Sales",
                "DistanceFromHome": 6,
                "Education": 2,
                "EducationField": "Life Sciences",
                "Gender": "Female",
                "JobLevel": 2,
                "JobRole": "Sales Executive",
                "MaritalStatus": "Single",
                "MonthlyIncome": 62000,
                "NumCompaniesWorked": 8,
                "PercentSalaryHike": 11,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 8,
                "TrainingTimesLastYear": 0,
                "YearsAtCompany": 6,
                "YearsSinceLastPromotion": 0,
                "YearsWithCurrManager": 5,
                "EnvironmentSatisfaction": 1,
                "JobSatisfaction": 4,
                "WorkLifeBalance": 1,
                "JobInvolvement": 3,
                "PerformanceRating": 3,
                "avg_work_hours": 9.6,
            }
        }
    }


class PredictionResponse(BaseModel):
    attrition_prediction: Literal["Yes", "No"]
    attrition_probability: float = Field(..., description="Predicted probability of attrition (class 1)")
    risk_level: Literal["Low", "Medium", "High"]
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    n_features: int
