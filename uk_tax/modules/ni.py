from enum import Enum
from .config import config
from typing import TypedDict, Optional
from .dataclasses import NationalInsurance


class NICategory(Enum):
    """
    National Insurance categories
    """
    A = "A"
    B = "B"
    C = "C"
    H = "H"
    

class NICalculationResult(TypedDict):
    """Result of National Insurance calculation"""
    standard_rate_ni: float
    additional_rate_ni: float
    total_ni: float
    rate_used: float
    additional_rate: float
    category: str

class NICalculator: 
    def __init__(self, category: NICategory, ni_data: Optional[NationalInsurance]=None):
        # Use provided NI data or get from config
        if ni_data is None:
            ni_data = config.national_insurance
        
        self.threshold = ni_data.primary_threshold
        self.limit = ni_data.upper_earnings_limit
        self.rates = ni_data.rates
        self.category = category
        
    def _get_Standard_rate(self):
        rates = self.rates
        
        match self.category:
            case NICategory.B:
                standard_rate = rates.reduced
            case NICategory.C:
                standard_rate = rates.over_pension_age
            case _:
                standard_rate = rates.standard
        
        return standard_rate
        
    def calculate_national_insurance(self, 
            gross_income: float)-> NICalculationResult:
        """
        Calculates national insurance payments based on category. 
        
        Param:
            gross_income: Annual Gross income
        Return:
            Dictionary of national insurance breakdown
        """
        
        allowance = self.threshold
        limit = self.limit
        standard_rate = self._get_Standard_rate()
        additional_rate = self.rates.additional
        
        if gross_income <= allowance:
            ni_standard = 0
            ni_additional = 0
        
        elif gross_income <= limit:
            ni_standard = (gross_income - allowance) * standard_rate
            ni_additional = 0
        else:
            ni_standard = (limit - allowance) * standard_rate
            ni_additional = (gross_income - limit) * additional_rate
        
        total_ni = ni_standard + ni_additional
        
        return {
            "standard_rate_ni": round(ni_standard, 2),
            "additional_rate_ni": round(ni_additional, 2),
            "total_ni": round(total_ni, 2),
            "rate_used": standard_rate,
            "additional_rate": additional_rate,
            "category": self.category.value
        }