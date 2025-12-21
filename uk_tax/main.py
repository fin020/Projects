from uk_tax.modules import UKTaxCalculator, StudentLoanPlan, NICategory

def main():
    calculator = UKTaxCalculator(
        student_loan_plan=StudentLoanPlan.Plan_5,
        ni_category=NICategory.A
    )
    
    print(calculator.generate_tax_summary(35000))

if __name__ == "__main__":
    main()