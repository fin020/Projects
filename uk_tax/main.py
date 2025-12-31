from .modules import UKTaxCalculator, StudentLoanPlan, NICategory

def main():
    calculator = UKTaxCalculator(
        student_loan_plan=StudentLoanPlan.NO_PLAN,
        ni_category=NICategory.A
    )
    
    print(calculator.generate_tax_summary(151000))

if __name__ == "__main__":
    main()