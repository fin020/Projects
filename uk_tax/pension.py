from enum import Enum
from typing import TypedDict, Optional
from uk_tax import config
from uk_tax.dataclasses import WorkplacePensionRates


class PensionScheme(Enum):
    WORKPLACE = "Workplace Pension Scheme"
    DB = "Defined Benefit Scheme"
    DC = "Defined Contribution Scheme"
    
class PensionResult(TypedDict):
    p_contribution: float
    e_contribution: float
    p_amount: float
    e_amount: float
    total: float
      

class PensionCalculator:
    def __init__(self, pension_data: Optional[WorkplacePensionRates]=None):
        if pension_data is None:
            pension_data = config.workplace_pension_rates
        
        self.personal_minimum = pension_data.personal_minimum
        self.employer_minimum = pension_data.employer_minimum    
        
    def pension_contributions(self, gross_income: float, p_contribution: float=0.0,
                                e_contribution: float =0.0) -> dict[str,float]:
        """
        Calculates the pension amounts for both employer and employee. 
        
        Params:
            gross_income: annual income
            p_contribution: Personal rate of contribution - defeault minimum
            e_contribution: Employer rate of contribution - default minimum
        
        Returns:
            Dictionary of outputs
        """
        if p_contribution == 0.0:
            p_contribution = self.personal_minimum
        if e_contribution == 0.0:
            e_contribution = self.employer_minimum
        
        p_amount = gross_income * p_contribution
        e_amount = gross_income * e_contribution
        total = p_amount + e_amount
        
        return {
            "personal_contribution_rate": p_contribution,
            "employer_contribution_rate": e_contribution,
            "personal_amount": p_amount,
            "employer_amount": e_amount,
            "total_amount": total
        }