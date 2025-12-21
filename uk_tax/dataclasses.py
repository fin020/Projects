from dataclasses import dataclass
from typing import Optional, List, TypedDict

@dataclass
class TaxBand:
    name: str
    rate: float
    lower_bound: float
    upper_bound: Optional[float]

class StudentLoanData(TypedDict):
    Plan_1: float
    Plan_2: float
    Plan_4: float
    Plan_5: float
    POSTGRADUATE: float

@dataclass
class NationalInsuranceRates:
    standard: float
    reduced: float
    additional: float
    over_pension_age: float

@dataclass
class NationalInsurance:
    primary_threshold: float
    upper_earnings_limit: float
    rates: NationalInsuranceRates

@dataclass
class WorkplacePensionRates:
    employer_minimum: float
    personal_minimum: float


@dataclass
class UKTaxConfig:
    version: str
    personal_allowance: float
    personal_allowance_taper_start: float
    personal_allowance_taper_rate: float
    tax_bands: List[TaxBand]
    national_insurance: NationalInsurance
    student_loan_thresholds: StudentLoanData
    student_loan_rates: StudentLoanData
    workplace_pension_rates: WorkplacePensionRates
