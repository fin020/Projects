from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any


class TaxYear(Enum):
    """ 
    Current Tax year for 2025-2026
    """
    YEAR_2025_2026 = "2025-2026"
    
class StudentLoanPlan(Enum):
    """
    Student Loan Plans
    """
    Plan_1 = "Plan 1"
    Plan_2 = "Plan 2"
    Plan_4 = "Plan 4"
    Plan_5 = "Plan 5"
    POSTGRADUATE = "Postgraduate"
    NO_PLAN = "No Plan"
    
@dataclass
class TaxBand:
    """
    Single tax and with rate and thresholds
    """
    name: str
    rate: float
    lower_bound: int
    upper_bound: Optional[int] = None


class NICategory(Enum):
    """
    National Insurance categories
    """
    A = "A"
    B = "B"
    C = "C"
    H = "H"

class UKTaxCalculator:
    """
    Docstring for UKTaxCalculator
    """
    TAX_YEAR_DATA:dict[Any] = {
        TaxYear.YEAR_2025_2026: {
            "personal_allowance": 12570,
            "personal_allowance_taper_start": 100000,
            "personal_allowance_taper_rate": 0.5,
            
            "tax_bands": [
                TaxBand("Basic Rate", 0.2, 0, 50270),
                TaxBand("Higher Rate", 0.4, 50270, 125140),
                TaxBand("Additional Rate", 0.45, 125140, None),
            ],
            
            "student_loan_thresholds": {
                StudentLoanPlan.Plan_1: 24295,
                StudentLoanPlan.Plan_2: 27295,
                StudentLoanPlan.Plan_4: 27660,
                StudentLoanPlan.Plan_5: 25000,
                StudentLoanPlan.POSTGRADUATE: 21000,
            },
            
            "student_loan_rates": {
                StudentLoanPlan.Plan_1: 0.09,
                StudentLoanPlan.Plan_2: 0.09,
                StudentLoanPlan.Plan_4: 0.09,
                StudentLoanPlan.Plan_5: 0.09,
                StudentLoanPlan.POSTGRADUATE: 0.06,
            },
        }
    }
    
    def __init__(self,
                 tax_year: TaxYear =TaxYear.YEAR_2025_2026,
                 student_loan_plan: StudentLoanPlan = StudentLoanPlan.NO_PLAN,
                 ni_category: NICategory = NICategory.A):
        self.tax_year = tax_year
        self.student_loan_plan = student_loan_plan
        self.ni_category = ni_category
        self.data = self.TAX_YEAR_DATA[tax_year]
        
    def calculate_personal_allowance(self, gross_income: float) -> float:
        """
        Docstring for calculate_personal_allowance
        
        :param self: Description
        :param gross_income: Description
        :type gross_income: float
        :return: Description
        :rtype: float
        """
        pa = self.data["personal_allowance"]
        taper_start = self.data["personal_allowance_taper"]
        if gross_income <= taper_start:
            return pa
        
        excess = gross_income - taper_start
        reduction = excess * self.data["personal_allowance_taper_rate"]
        adjusted_pa = max(0, pa - reduction)
        
        return adjusted_pa
    
    def student_loan_payable(self, gross_income):
        if (self.student_loan_plan == StudentLoanPlan.NO_PLAN or
            gross_income <= 0):
            return {"annual_repayment": 0.0, "monthly_repayment": 0.0}
        
        threshold = self.data["student_loan_thresholds"][self.student_loan_plan]
        rate = self.data["student_loan_rates"][self.student_loan_plan]
        
        if gross_income <= threshold:
            repayment = 0.0        
        else:
            repayment = (gross_income - threshold) * rate
            
        return {"annual_repayment": round(repayment, 2),
                "monthly_repayment": round(repayment/12, 2),
                "threshold":threshold, 
                "Rate": rate}    
    
    def calculate_income_tax(self, gross_income: float) -> dict[str, float]:
        """
        Docstring for calculate_income_tax
        
        :param self: Description
        :param gross_income: Description
        :type gross_income: float
        :return: Description
        :rtype: dict[str, float]
        """
        
        personal_allowance = self.calculate_personal_allowance(gross_income)
        taxable_income = max(0, gross_income - personal_allowance)
        
        tax_bands = self.data["tax_bands"]
        tax_details = {}
        total_tax = 0.0
        remaining_income = taxable_income
        
        for band in tax_bands
        
        