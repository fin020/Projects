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

@dataclass
class TaxInfo:
    personal_allowance: float
    taxable_income: float
    tax_details: dict[str, float]
    total_tax: float
    effective_tax_rate: float

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
    TAX_YEAR_DATA: dict[TaxYear, dict[str, Any]] = {
        TaxYear.YEAR_2025_2026: {
            "personal_allowance": 12570,
            "personal_allowance_taper_start": 100000,
            "personal_allowance_taper_rate": 0.5,
            
            "tax_bands": [
                TaxBand("Basic Rate", 0.2, 0, 50270),
                TaxBand("Higher Rate", 0.4, 50270, 125140),
                TaxBand("Additional Rate", 0.45, 125140, None),
            ],
            
            "national_insurance": {
                "primary_threshold": 12570,
                "upper_earnings_limit": 50270,
                "rates": {
                    "standard": 0.08,
                    "reduced": 0.0585,
                    "additional": 0.02,
                    "over_pension_age": 0.0
                }
            },
            
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
        taper_start = self.data["personal_allowance_taper_start"]
        if gross_income <= taper_start:
            return pa
        
        excess = gross_income - taper_start
        reduction = excess * self.data["personal_allowance_taper_rate"]
        adjusted_pa = max(0, pa - reduction)
        
        return adjusted_pa
    
    def student_loan_payable(self, gross_income: float) -> dict[str, float]:
        if self.student_loan_plan == StudentLoanPlan.NO_PLAN or gross_income <= 0:
            return {"annual_repayment": 0.0, "monthly_repayment": 0.0, "threshold": 0.0, "Rate": 0.0}
        
        threshold = float(self.data["student_loan_thresholds"][self.student_loan_plan])
        rate = float(self.data["student_loan_rates"][self.student_loan_plan])
        
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
    
    def calculate_income_tax(self, gross_income: float) -> TaxInfo:
        """
        Docstring for calculate_income_tax
        
        :param self: Description
        :param gross_income: Description
        :type gross_income: float
        :return: Description
        :rtype: dict[str, float]
        """
        
        personal_allowance = self.calculate_personal_allowance(gross_income)
        taxable_income = max(0, gross_income - personal_allowance - self.student_loan_payable(gross_income)["annual_repayment"])
        tax_bands = self.data["tax_bands"]
        tax_details: dict[str, float] = {}
        total_tax = 0.0
        remaining_income = taxable_income
        
        for band in tax_bands:
            if remaining_income <= 0:
                tax_details[band.name] = 0
                continue
            
            band_width = (band.upper_bound - band.lower_bound
                          if band.upper_bound else float('inf'))
            
            taxable_in_band = min(remaining_income, band_width)
            tax_in_band = taxable_in_band * band.rate
            tax_details[band.name] = round(tax_in_band, 2)
            total_tax += tax_in_band
            remaining_income -= taxable_in_band
            
        effective_tax_rate = total_tax / gross_income * 100 if gross_income > 0 else 0.0
            
        return TaxInfo(personal_allowance=personal_allowance,
                       taxable_income=taxable_income,
                       tax_details=tax_details,
                       total_tax=total_tax,
                       effective_tax_rate=effective_tax_rate)
    def calculate_national_insurance(self, gross_income: float)-> dict[str, float]:
        """
        Docstring for calculate_national_insurance
        
        :param self: Description
        :param gross_income: Description
        """
        allowance = self.data["national_insurance"]["primary_threshold"]
        limit = self.data["national_insurance"]["upper_earnings_limit"]
        rates = self.data["national_insurance"]["rates"]
        
        match self.ni_category:
            case NICategory.B:
                standard_rate = rates["reduced"]
            case NICategory.C:
                standard_rate = rates["over_pension_age"]
            case _:
                standard_rate = rates["standard"]
        
        additional_rate = rates["additional"]
        
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
            "standard_rate_ni": round(ni_standard,2),
            "additional_rate_ni": round(ni_additional,2),
            "total_ni": round(total_ni,2),
            "rate_used": standard_rate,
            "additional_rate": additional_rate
        }
    
    def calculate_take_home_pay(self, gross_income:float) -> dict[str, float]:
        """
        Docstring for calculate_takehome_pay
        
        :param self: Description
        :param gross_income: Description
        :type gross_income: float
        """
        student_loan_payment = float(self.student_loan_payable(gross_income).get("annual_repayment", 0.0))
        national_insurance = float(self.calculate_national_insurance(gross_income).get("total_ni", 0.0))
        tax_information = self.calculate_income_tax(gross_income)
        
        tax_payable = tax_information.total_tax
        
        deductions: float = student_loan_payment + national_insurance + tax_payable
        net_income = gross_income - deductions
        
        return {
            "gross_income": gross_income,
            "income_tax": tax_payable,
            "national_insurance": national_insurance,
            "student_loan": student_loan_payment,
            "total_deductions": deductions,
            "net_income": net_income,
            "monthly_take_home": round(net_income / 12 , 2),
            "weekly_take_home": round(net_income / 52, 2)
        }
    
    def generate_tax_summary(self, gross_income: float): 
        """
        Docstring for generate_tax_summary
        
        :param self: Description
        :param gross_income: Description
        :type gross_income: float
        """
        result = self.calculate_take_home_pay(gross_income)
        taxes = self.calculate_income_tax(gross_income)
        ni = self.calculate_national_insurance(gross_income)
        sl = self.student_loan_payable(gross_income)
        
        summary = [
            '=' * 50,
            f"UK TAX CALCULATOR - {self.tax_year.value}",
            '=' * 50,
            f"\nAnnual Gross Salary: £{result['gross_income']:,.2f}",
            "\nINCOME TAX BREAKDONW:",
            f"  Personal Allowance: £{taxes.personal_allowance:,.2f}",
            f"  Taxable Income: £{taxes.taxable_income:,.2f}",
        ]
        
        for band, amount in taxes.tax_details.items():
            if amount > 0:
                summary.append(f"    {band}: £{amount:,.2f}")
                
        summary.extend([
            f"  Total Income Tax: £{taxes.total_tax:,.2f}",
            f"  Effective Tax Rate: £{taxes.effective_tax_rate:,.2f}",
            f"\nNATIONAL INSURANCE",
            f"  Standard Rate (£{ni['standard_rate_ni']:,.2f} at {ni['rate_used']*100}%)",
            f"  Additional Rate (£{ni['additional_rate_ni']:,.2f} at {ni['additional_rate']*100}%)",
            f"  Total NI: £{ni["total_ni"]:,.2f}"
        ])
        
        if self.student_loan_plan != StudentLoanPlan.NO_PLAN:
            summary.extend([
                f"\nSTUDENT LOAN ({self.student_loan_plan.value}):",
                f"  Annual Repayment: £{sl["annual_repayment"]:,.2f}",
                f"  Monthly: £{sl["monthly_repayment"]:,.2f}"
            ])
            
        summary.extend([
            "\nSUMMARY:",
            F"  Total Deductions: £{result["total_deductions"]:,.2f}",
            F"  Annual Take Home: £{result["net_income"]:,.2f}",
            F"  Monthly Take Home: £{result["monthly_take_home"]:,.2f}",
            F"  Weekly Take Home: £{result["weekly_take_home"]:,.2f}"
        ])
        
        return '\n'.join(summary)
    
if __name__ == "__main__":
    myCalculator = UKTaxCalculator(
        student_loan_plan=StudentLoanPlan.Plan_5,
        ni_category=NICategory.A)
    
    print(myCalculator.generate_tax_summary(45000))
    