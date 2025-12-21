from uk_tax import config
from ni import NICalculator, NICategory
from pension import PensionCalculator
from student_loan import StudentLoanCalculator, StudentLoanPlan
from uk_tax.dataclasses import UKTaxConfig

class IncomeTax():
    def __init__(self, ni_category:NICategory, student_loan_plan:StudentLoanPlan,
                tax_data: UKTaxConfig | None =None):
        if tax_data is None:
            tax_data = config
        self.personal_allowance = tax_data.personal_allowance
        self.tax_bands = tax_data.tax_bands
        self.taper_start = tax_data.personal_allowance_taper_start
        self.taper_rate =tax_data.personal_allowance_taper_rate
        self.ni = NICalculator(category=ni_category)
        self.pension = PensionCalculator()
        self.student_loan = StudentLoanCalculator(plan=student_loan_plan)
        
        
    def calculate_personal_allowance(self, gross_income: float) -> float:
            """
            Calculates personal tax allowance. Adjusts for high earners. 
            For every £2 earned above £100,000, personal allowance is reduced by £1
            
            Args:
                gross_income: annually 
            
            Returns:
                Adjusted personal allowance annually
            """
            pa = self.personal_allowance
            taper_start = self.taper_start
            taper_rate = self.taper_rate
            pension_amount = self.pension.pension_contributions(gross_income=gross_income)['personal_amount']
            income_pension = gross_income - pension_amount
            if income_pension <= taper_start:
                return pa
            
            excess = income_pension - taper_start
            reduction = excess * taper_rate
            adjusted_pa = max(0, pa - reduction)
            
            return adjusted_pa
        
    def calculate_income_tax(self, gross_income: float) -> dict[str, float | dict[str,float]]:
            """
            Calculates income tax paid
            
            Param:
                gross_income: Annual Gross income
            :return: 
                Dataclass with income tax breakdown. 
            """
            pension =self.pension.pension_contributions(gross_income=gross_income)
            pension_reduction = pension['personal_amount']
            personal_allowance = self.calculate_personal_allowance(gross_income=gross_income)
            loan = self.student_loan.student_loan_payable(gross_income)
            
            taxable_income = max(0, gross_income - personal_allowance - loan.get('annual_repayment', 0.0) - pension_reduction)
            tax_bands = self.tax_bands
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
                
            return {"personal_allowance": personal_allowance,
                        "taxable_income": taxable_income,
                        "tax_details": tax_details,
                        "total_tax": total_tax,
                        "effective_tax_rate": effective_tax_rate}
            

    def calculate_take_home_pay(self, gross_income: float) -> dict[str, float]:
        """
        Calcuates take home pay based on gross income and all deductions.
        Deductions include: tax, national insurance and student loan payments
        
        Param: 
            gross_income: Annual Gross income
        Return: 
            Dictionary of breakdown of take home pay
        """
        student_loan_payment = self.student_loan.student_loan_payable(gross_income).get("annual_repayment", 0.0)
        national_insurance = float(self.ni.calculate_national_insurance(gross_income).get("total_ni", 0.0))
        tax_information = self.calculate_income_tax(gross_income)
        pension = self.pension.pension_contributions(gross_income)
        
        tax_payable = tax_information.get("total_tax")
        if not isinstance(tax_payable, float):
            tax_payable = 0.0
        
        deductions: float = student_loan_payment + national_insurance + tax_payable + pension['personal_amount']
        net_income = gross_income - deductions
        
        return {
            "gross_income": gross_income,
            "income_tax": tax_payable,
            "national_insurance": national_insurance,
            "student_loan": student_loan_payment,
            "pension_amount": pension['personal_amount'],
            "total_deductions": deductions,
            "net_income": net_income,
            "monthly_take_home": round(net_income / 12 , 2),
            "weekly_take_home": round(net_income / 52, 2)
        }