from enum import Enum
from uk_tax import config


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
    

class StudentLoanCalculator:
    def __init__(self, plan: StudentLoanPlan):
        self.plan = plan
        self.thresholds = config.student_loan_thresholds
        self.rates = config.student_loan_rates
        
    def _get_plan_key(self) -> str:
        match self.plan:
            case StudentLoanPlan.Plan_1:
                return "Plan_1"
            case StudentLoanPlan.Plan_2:
                return "Plan_2"
            case StudentLoanPlan.Plan_4:
                return "Plan_4"
            case StudentLoanPlan.Plan_5:
                return "Plan_5"
            case StudentLoanPlan.POSTGRADUATE:
                return "POSTGRADUATE"
            case _:
                raise ValueError(f"Unknown Plan: {self.plan}")
        
    
    def student_loan_payable(self, gross_income: float) -> dict[str, float]:
        """
        Calculates student loan to be paid annually for each student loan plan.
        
        Param:
            gross_income: Annual Gross income
        
        Return: 
            Dictionary with Annual, monthly payments and the rate and threshold. 
        """
        if self.plan == StudentLoanPlan.NO_PLAN or gross_income <= 0:
            return {"annual_repayment": 0.0, "monthly_repayment": 0.0, "threshold": 0.0, "Rate": 0.0}
        
        plan_key = self._get_plan_key()
        threshold = self.thresholds.get(plan_key,0)
        rate = self.rates.get(plan_key,0)
        if gross_income <= threshold:
            repayment = 0.0        
        else:
            repayment = (gross_income - threshold) * rate
            
        return {
            "annual_repayment": round(repayment, 2),
            "monthly_repayment": round(repayment / 12, 2),
            "threshold": threshold,
            "rate": rate
        }