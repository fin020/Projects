from . import config
from .dataclasses import UKTaxConfig
from typing import Optional
from .ni import NICategory, NICalculator
from .income_tax import IncomeTax
from .student_loan import StudentLoanCalculator, StudentLoanPlan
from .pension import PensionCalculator


class UKTaxCalculator:
    """
    UK Tax calculator for 2025-2026. 
    
    Calculates Income Tax, National Insurance and student Loan repayments.
    """
    def __init__(self, data: Optional[UKTaxConfig]|None=None,
                 student_loan_plan: StudentLoanPlan = StudentLoanPlan.NO_PLAN,
                 ni_category: NICategory = NICategory.A):
        """
        Initialises the tax calculator
        
        Args:
        tax_year: Tax year used  in calculations
        student_loan_plan: Repayment plan
        ni_category: National Insurance Category
        """
        if data is None: 
            data = config.config
        self.tax_year = data.version
        self.student_loan_plan = student_loan_plan
        self.ni_category = ni_category
        self.pension = PensionCalculator()
        self.ni = NICalculator(category=ni_category)
        self.student_loan = StudentLoanCalculator(plan=student_loan_plan)
        self.tax = IncomeTax(ni_category=ni_category, 
                             student_loan_plan=student_loan_plan)

    def generate_tax_summary(self, gross_income: float): 
        """
        
        Params: 
            gross_income: Annual Income
            
        Return:
            Returns summary of income
        """
        result = self.tax.calculate_take_home_pay(gross_income)
        taxes = self.tax.calculate_income_tax(gross_income)
        ni = self.ni.calculate_national_insurance(gross_income)
        sl = self.student_loan.student_loan_payable(gross_income)
        pension = self.pension.pension_contributions(gross_income)
        
        summary = [
            '=' * 50,
            f"UK TAX CALCULATOR - {self.tax_year}",
            '=' * 50,
            f"\nAnnual Gross Salary: £{result['gross_income']:,.2f}",
            "\nINCOME TAX BREAKDONW:",
            f"  Personal Allowance: £{taxes.get('personal_allowance'):,.2f}",
            f"  Taxable Income: £{taxes.get('taxable_income'):,.2f}",
        ]
        tax_details = taxes.get('tax_details')
        if isinstance(tax_details, dict):
            for band, amount in tax_details.items():
                if amount > 0:
                    summary.append(f"    {band}: £{amount:,.2f}")
                
        summary.extend([
            f"  Total Income Tax: £{taxes.get('total_tax'):,.2f}",
            f"  Effective Tax Rate: {taxes.get('effective_tax_rate'):,.2f}%",
            f"\nNATIONAL INSURANCE:",
            f"  Standard Rate: (£{ni['standard_rate_ni']:,.2f} at {ni['rate_used']*100}%)",
            f"  Additional Rate: (£{ni['additional_rate_ni']:,.2f} at {ni['additional_rate']*100}%)",
            f"  Total NI: £{ni["total_ni"]:,.2f}"
        ])
        
        if self.student_loan_plan != StudentLoanPlan.NO_PLAN:
            summary.extend([
                f"\nSTUDENT LOAN ({self.student_loan_plan.value}):",
                f"  Annual Repayment: £{sl['annual_repayment']:,.2f}",
                f"  Monthly: £{sl['monthly_repayment']:,.2f}"
            ])
            
        summary.extend([
            "\nPension Breakdown:",
            f"  Personal Contribution: £{pension['personal_amount']:,.2f} at {pension['personal_contribution_rate']*100}% ",
            f"  Employer Contribution: £{pension['employer_amount']:,.2f} at {pension['employer_contribution_rate']*100}% ",
            f"  Total amount: £{pension['total_amount']:,.2f}"
        ])
            
        summary.extend([
            "\nSUMMARY:",
            f"  Total Deductions: £{result['total_deductions']:,.2f}",
            f"  Annual Take Home: £{result['net_income']:,.2f}",
            f"  Monthly Take Home: £{result['monthly_take_home']:,.2f}",
            f"  Weekly Take Home: £{result['weekly_take_home']:,.2f}",
            f"  Deduction rate: {result['total_deductions'] / gross_income * 100:.2f}%"
        ])
        
        return '\n'.join(summary)
    
