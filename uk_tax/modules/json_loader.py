import json
from typing import cast
from .dataclasses import (
    UKTaxConfig,
    TaxBand,
    NationalInsurance,
    NationalInsuranceRates,
    WorkplacePensionRates,
    StudentLoanData
)

def load_tax_config(path: str) -> UKTaxConfig:
    with open(path, "r") as f:
        raw = json.load(f)

    tax_bands = [TaxBand(**tb) for tb in raw["tax_bands"]]

    ni_rates = NationalInsuranceRates(
        **raw["national_insurance"]["rates"]
    )

    ni = NationalInsurance(
        primary_threshold=raw["national_insurance"]["primary_threshold"],
        upper_earnings_limit=raw["national_insurance"]["upper_earnings_limit"],
        rates=ni_rates
    )

    workplace_pension = WorkplacePensionRates(
        **raw["workplace_pension_rates"]
    )
    
    student_loan_thresholds = cast(StudentLoanData, raw["student_loan_thresholds"])
    student_loan_rates = cast(StudentLoanData, raw["student_loan_rates"])

    return UKTaxConfig(
        version=raw["version"],
        personal_allowance=raw["personal_allowance"],
        personal_allowance_taper_start=raw["personal_allowance_taper_start"],
        personal_allowance_taper_rate=raw["personal_allowance_taper_rate"],
        tax_bands=tax_bands,
        national_insurance=ni,
        student_loan_thresholds=student_loan_thresholds,
        student_loan_rates=student_loan_rates,
        workplace_pension_rates=workplace_pension
    )
