
from .dataclasses import *
from .tax_calculator import UKTaxCalculator
from .income_tax import IncomeTax
from .ni import NICalculator, NICategory
from .student_loan import StudentLoanCalculator, StudentLoanPlan
from .pension import PensionCalculator

__all__ = [
    'UKTaxCalculator',
    'IncomeTax',
    'NICalculator',
    'NICategory',
    'StudentLoanCalculator',
    'StudentLoanPlan',
    'PensionCalculator',
]

# Get the correct path

version = "1.0.0"
print(f"Welcome to my tax calculator {version}")