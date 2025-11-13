from abc import ABC, abstractmethod
from datetime import date, datetime
from matplotlib import pyplot as plt
from typing import Optional, Any
import numpy as np
import pandas as pd
from pandas import DataFrame

plt.style.use("classic")


#interest rates:
RPI = 0.032 #annual
SALARY_INCREASE_RATE = 0.03 #annual
EARLY_REPAYMENT = 1000 #annual


class StudentLoanPlan(ABC):
    """Base class for student loan plans with common functionality."""
    def __init__(self, balance:float, salary:float, year_end:date):
        """
        Initialise loan plan.
        
        Args:
            balance (float): Starting loan balance in £
            salary (float): Annual salary in £
            year_end (date): year when loan repayments start
        """
        self.balance = balance
        self.salary = salary
        self.year_end = year_end
        self.first_repayment_date = datetime(year_end.year + 1, 4, 1) if year_end.month > 4 else datetime(year_end.year, 4, 1)
        
    @staticmethod
    def get_plan(balance: float, salary: float, year_end: date) -> 'StudentLoanPlan':
        """
        Factory method to create appropriate loan plan based on start year.
        
        Args:
            balance (float): Starting loan balance
            salary (float): Annual salary
            year_end (date): Loan start date
            
        Returns:
            StudentLoanPlan: Appropriate plan instance (Plan1, Plan2, or Plan5)
        """
        if year_end.year < 2012:
            return Plan1Loan(balance, salary, year_end)
        elif 2012 <= year_end.year < 2022:
            return Plan2Loan(balance, salary, year_end)
        else:
            return Plan5Loan(balance, salary, year_end)
             
    @abstractmethod
    def interest_rate(self, salary:Optional[float] = None) -> float:
        """return interest rate based on current RPI conditions and potential inflation"""
        pass 
    @abstractmethod
    def repayment_threshold(self) -> int: 
        """returns salary threshold for repayment in £"""
        pass
    @abstractmethod
    def repayment_rate(self)-> float:
        """returns the % of salary above the threshold that goes towards repayment"""
        pass
    @abstractmethod
    def loan_writeoff(self)-> int:
        """Returns years the loan is written off"""
        pass
    
    def annual_repayment(self,salary:float | None = None, balance: float | None = None) -> float:
        """
        Calculate annual repayment amount.
        
        Args:
            salary (float, optional): Annual salary. Defaults to self.salary.
            balance (float, optional): Current balance. Defaults to self.balance.
            
        Returns:
            float: Annual repayment amount in £
        """
        salary = salary if salary is not None else self.salary
        balance = balance if balance is not None else self.balance
        
        if balance < 0.01:
            return 0.0
        
        threshold = self.repayment_threshold()
        rate = self.repayment_rate()
        return max(0.0, (salary - threshold) * rate)
    
    def monthly_payment(self, salary:float | None = None,
                        balance: float | None = None)-> float:
        """
        Calculate monthly repayment amount.
        
        Args:
            salary (float, optional): Monthly salary. If None, uses annual/12.
            balance (float, optional): Current balance. Defaults to self.balance.
            
        Returns:
            float: Monthly repayment amount in £
        """
        annual_salary = salary if salary is not None else self.salary
        balance = balance if balance is not None else self.balance
        return self.annual_repayment(annual_salary, balance) / 12
    
    def monthly_interest_rate(self, salary: float| None = None) -> float:
        """
        Convert annual interest rate to effective monthly rate.
        
        Uses formula: (1 + r_annual)^(1/12) - 1
        
        Args:
            salary (float, optional): Annual salary. Defaults to self.salary.
            
        Returns:
            float: Monthly interest rate as decimal
        """
        salary = salary if salary is not None else self.salary
        annual_rate = self.interest_rate(salary)
        return np.power(1 + annual_rate, 1/12) - 1
    
    def forecast(self, early_repayment: float = EARLY_REPAYMENT) -> list[dict[int, tuple[float, float, float, float]]]:
        """
        Generate annual forecast of loan balance, repayments, and salary.
        
        This uses monthly compounding internally for accuracy, then aggregates to annual results.
        
        Args:
            early_repayment (float): Additional annual repayment amount in £
            
        Returns:
            list[dict]: List of dicts with year as key and tuple 
                       (balance, repayment, salary, interest_rate) as value
        """
        current_year = self.first_repayment_date.year if self.first_repayment_date > datetime.now() else datetime.now().year
        current_month = self.first_repayment_date.month if self.first_repayment_date > datetime.now() else datetime.now().month
        
        # Calculate writeoff date: April + loan_writeoff years from first repayment
        writeoff_year = self.first_repayment_date.year + self.loan_writeoff()
        writeoff_month = 4  # April
        
        # Calculate total months from now until writeoff
        years_diff = writeoff_year - current_year
        months_diff = writeoff_month - current_month
        total_months = years_diff * 12 + months_diff
        
        balance = self.balance
        annual_salary = self.salary
        monthly_early = early_repayment / 12
        
        # Track annual aggregates
        current_forecast_year = current_year
        annual_repayment_sum = 0.0
        month_in_year = current_month
        annual_early_sum = 0.0
        forecast = []
        
        for month in range(total_months):
            # Calculate current month and year
            current_month_num = (current_month + month - 1) % 12 + 1
            current_year_num = current_year + (current_month + month - 1) // 12
            
            if current_month_num == 4 and month > 0:
                annual_salary *= (1 + SALARY_INCREASE_RATE)
            
            # Calculate monthly values
            monthly_interest_rate = self.monthly_interest_rate(annual_salary)
            monthly_interest = balance * monthly_interest_rate
            
            monthly_repayment = self.monthly_payment(annual_salary, balance)
            annual_repayment_sum += monthly_repayment
            annual_early_sum += monthly_early

            total_monthly_payment = monthly_repayment + monthly_early
            if total_monthly_payment > balance + monthly_interest:
                actual_payment = balance + monthly_interest
                monthly_repayment_adjusted = min(monthly_repayment, actual_payment)
                monthly_early_adjusted = actual_payment - monthly_repayment_adjusted
                
                annual_repayment_sum -= (monthly_repayment - monthly_repayment_adjusted)
                annual_early_sum -= (monthly_early - monthly_early_adjusted)
                
                balance = 0
            else: 
                balance += monthly_interest - total_monthly_payment
                
                
            # Increment month
            month_in_year += 1
            
            # At year end (or when balance hits zero), record annual data
            if current_month_num == 12 or balance == 0 or month == total_months:
                annual_total_payment = annual_repayment_sum + annual_early_sum
                
                forecast.append({ #type: ignore
                    current_forecast_year: (
                        round(balance, 2),
                        round(annual_total_payment, 2),
                        round(annual_salary, 2),
                        round(self.interest_rate(annual_salary), 4)
                    )
                })
                
                # Reset for next year
                annual_repayment_sum = 0.0 
                annual_early_sum = 0.0
                current_forecast_year = current_year_num + 1
                if balance <= 0:
                    break
        
        return forecast #type: ignore
    
    def monthly_forecast(self, early_repayment: float = EARLY_REPAYMENT) -> list[dict[int,tuple[float,float,float,float]]]:
        """
        Generate monthly forecast of loan balance, repayments, and salary.
        
        Args:
            early_repayment (float): Additional annual repayment (divided by 12 monthly)
            
        Returns:
            list[dict]: List of dicts with YYYYMM as key and tuple 
                       (balance, repayment, salary, interest_rate) as value
        """
        current_year = self.first_repayment_date.year if self.first_repayment_date > datetime.now() else datetime.now().year
        current_month = self.first_repayment_date.month if self.first_repayment_date > datetime.now() else datetime.now().month
        
        # Calculate months to simulate - from now until writeoff year
        writeoff_year = self.first_repayment_date.year + self.loan_writeoff()  # From April after course
        writeoff_month = 4
        total_months = (writeoff_year - current_year) * 12 + (writeoff_month - current_month)
        
        balance = self.balance
        annual_salary = self.salary
        monthly_early = early_repayment / 12
        
        current_yyyymm = current_year * 100 + current_month
        
        forecast = [{
            current_yyyymm: (
                round(self.balance, 2),
                round(self.monthly_payment(), 2),
                round(self.salary / 12, 2),
                round(self.monthly_interest_rate(), 4)
            )
        }]
        
        # Track which month we're in for annual salary increases
        month_counter = current_month
        
        for month_idx in range(1, total_months): #type: ignore
            # Calculate monthly values
            monthly_interest_rate = self.monthly_interest_rate(annual_salary)
            monthly_interest = balance * monthly_interest_rate
            monthly_repayment = self.monthly_payment(annual_salary, balance) + monthly_early
            
            # Update balance
            balance = balance + monthly_interest - monthly_repayment
            
            # If overpaid, adjust
            if balance < 0:
                monthly_repayment += balance
                balance = 0
            
            # Increment month
            month_counter += 1
            if month_counter > 12:
                month_counter = 1
                # Apply annual salary increase in January
                annual_salary *= (1 + SALARY_INCREASE_RATE)
            
            # Calculate next YYYYMM
            year_part = current_yyyymm // 100
            month_part = (current_yyyymm % 100) + 1
            if month_part > 12:
                month_part = 1
                year_part += 1
            current_yyyymm = year_part * 100 + month_part
            
            forecast.append({
                current_yyyymm: (
                    round(balance, 2),
                    round(monthly_repayment, 2),
                    round(annual_salary / 12, 2),
                    round(monthly_interest_rate, 4)
                )
            })
            
            if balance == 0:
                break
        
        return forecast
        
    def forecast_to_dataframe(self, monthly: bool = False, 
                             early_repayment: float = EARLY_REPAYMENT) -> DataFrame:
        """
        Convert forecast to pandas DataFrame for easier analysis.
        
        Args:
            monthly (bool): If True, use monthly forecast; else annual
            early_repayment (float): Additional repayment amount
            
        Returns:
            pd.DataFrame: DataFrame with columns [period, balance, repayment, salary, interest_rate]
        """  
        forecast_data = (self.monthly_forecast(early_repayment) if monthly 
                        else self.forecast(early_repayment))
        
        # Extract data
        periods:list[int] = []
        balances:list[float] = []
        repayments:list[float] = []
        salaries:list[float] = []
        interest_rates:list[float] = []
        
        for d in forecast_data:
            try:
                key = list(d.keys())[0]
                values = list(d.values())[0]
                
                periods.append(key)
                balances.append(float(values[0]))  # Ensure float type
                repayments.append(float(values[1]))
                salaries.append(float(values[2]))
                interest_rates.append(float(values[3]))
            except (IndexError, TypeError) as e: #type: ignore
                continue
        
        df = pd.DataFrame({
            'period': periods,
            'balance': balances,
            'repayment': repayments,
            'salary': salaries,
            'interest_rate': interest_rates
        }).set_index('period')
        
        return df
        
    
    def forecast_plot(self):
        """Plots the forecast of repayments and loan balance"""
        forecast_data = self.forecast()
        fig, ax= plt.subplots() #type:ignore
        years = [list(d.keys())[0] for d in forecast_data]           # [2020, 2021, 2022]
        balances = [list(d.values())[0][0] for d in forecast_data] 
        repayments = [list(d.values())[0][1] for d in forecast_data]# extract balance (second element)
        ax.plot(years, balances, linewidth=2, label="Loan Balance") #type:ignore
        ax.plot(years, repayments,linewidth=2, label="Annual Repayments") #type: ignore
        ax.legend(loc="upper right") #type: ignore
        ax.set_xlabel("Year") #type:ignore
        ax.get_xaxis().get_major_formatter().set_useOffset(False)  #type: ignore
        ax.grid(True) #type: ignore
        ax.set_ylabel("Loan Balance (£)", fontweight="bold") #type:ignore
        ax.set_title("Forecast of Student Loan Balance", fontweight="bold") #type:ignore
        plt.style.use("classic")
        fig.tight_layout()
        fig.set_figheight(15)
        return plt.show() #type:ignore
    
    def monthly_forecast_plot(self):
        """Plots the forecast of repayments and loan balance"""
        forecast_data = self.monthly_forecast()
        fig, ax= plt.subplots() #type:ignore
        years = [list(d.keys())[0] for d in forecast_data]           # [2020, 2021, 2022]
        balances = [list(d.values())[0][0] for d in forecast_data] 
        repayments = [list(d.values())[0][1] for d in forecast_data]# extract balance (second element)
        ax.plot(years, balances, linewidth=2, label="Loan Balance") #type:ignore
        ax.plot(years, repayments,linewidth=2, label="Annual Repayments") #type: ignore
        ax.legend(loc="upper right") #type: ignore
        ax.set_xlabel("Year", fontweight="bold") #type:ignore
        ax.get_xaxis().get_major_formatter().set_useOffset(False)  #type: ignore
        ax.grid(True) #type: ignore
        ax.set_ylabel("Loan Balance (£)", fontweight="bold") #type:ignore
        ax.set_title("Forecast of Student Loan Balance", fontweight="bold") #type:ignore
        plt.style.use("classic")
        fig.tight_layout()
        fig.set_figheight(15)
        return plt.show() #type:ignore
    
    def total_lifetime_repayment(self, early_repayment: float = EARLY_REPAYMENT) -> float:
        """
        Calculate total amount repaid over loan lifetime.
        
        Args:
            early_repayment (float): Additional annual repayment
            
        Returns:
            float: Total lifetime repayment in £
        """
        forecast_data = self.monthly_forecast(early_repayment)
        total_repayment = 0.0
        
        for month_data in forecast_data:
            month_repayment = list(month_data.values())[0][1]  # repayment amount
            total_repayment += month_repayment
        
        return round(total_repayment, 2)
    
    def present_value_of_repayments(self, early_repayment: float = EARLY_REPAYMENT,
                                   discount_rate: float = RPI) -> float:
        """
        Calculate present value of all lifetime repayments.
        
        Uses discount formula: PV = FV / (1 + r)^t
        
        Args:
            early_repayment (float): Additional annual repayment
            discount_rate (float): Annual discount rate (default: RPI)
            
        Returns:
            float: Present value of repayments in £
        """
        df = self.forecast_to_dataframe(monthly=True, early_repayment=early_repayment)
        
        if df.empty:
            return 0 
        
        n_months = np.arange(len(df))

        
        discount_factors = np.power(1 + discount_rate, n_months / 12)
        present_values = df['repayment'].to_numpy() / discount_factors #type: ignore
        
        return round(float(np.sum(present_values)), 2) #type: ignore
    
    def present_value_plot(self, max_early_repayment: float = 5000, discount_rate: float = RPI):
        """
        Plot present value and total repayment against early repayment amounts.
        
        Args:
            max_early_repayment (float): Maximum early repayment to simulate
            discount_rate (float): Discount rate for present value calculation
        """
        early_repayments = np.linspace(0, max_early_repayment, 500)
        
        original_forecast_method = self.forecast_to_dataframe
        
        def silent_forecast_to_dataframe(*args:Any, **kwargs:Any):
            # Temporarily remove debug prints
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                result = original_forecast_method(*args, **kwargs)
            finally:
                sys.stdout = old_stdout
            return result
        
        # Apply the silent method temporarily
        self.forecast_to_dataframe = silent_forecast_to_dataframe
        
        try:
            present_values = [self.present_value_of_repayments(er, discount_rate) for er in early_repayments]
            total_values = [self.total_lifetime_repayment(er) for er in early_repayments]
            
            fig, ax = plt.subplots(figsize=(12, 6)) #type: ignore
            
            ax.plot(early_repayments, total_values, linewidth=2, #type: ignore
                label='Total Lifetime Repayment', alpha=0.7)
            ax.plot(early_repayments, present_values, linewidth=2, #type: ignore 
                label=f'Present Value (r={discount_rate:.1%})')
            
            ax.set_xlabel('Additional Annual Repayment (£)', fontweight='bold') #type: ignore
            ax.set_ylabel('Total Amount (£)', fontweight='bold') #type: ignore
            ax.set_title('Impact of Early Repayments on Total Cost', #type: ignore
                        fontweight='bold', fontsize=14)
            ax.legend() #type: ignore
            ax.grid(True, alpha=0.3) #type: ignore
            plt.tight_layout()
            return plt.show() #type: ignore
        
        finally:
            # Restore original method
            self.forecast_to_dataframe = original_forecast_method


        
        


class Plan1Loan(StudentLoanPlan):
    """Plan 1 loans: Started before 2012."""
    
    def interest_rate(self, salary: Optional[float] = None) -> float:
        return RPI
    
    def repayment_threshold(self) -> int:
        return 26065
    
    def repayment_rate(self) -> float:
        return 0.09
    
    def loan_writeoff(self) -> int:
        return 25

    

class Plan2Loan(StudentLoanPlan):
    """Plan 2 loans: Started 2012-2021. Interest varies with salary."""
    
    def interest_rate(self, salary: Optional[float] = None) -> float:
        """
        Calculate interest rate based on salary bands.
        
        - Below £28,470: RPI only
        - £28,470 - £51,245: RPI + (0% to 3% progressive)
        - Above £51,245: RPI + 3%
        """
        salary = salary if salary is not None else self.salary
        
        if salary < 28470:
            return RPI
        elif salary < 51245:
            # Progressive rate between RPI and RPI+3%
            differential = (salary - 28470) / (51245 - 28470)
            return RPI + differential * 0.03
        else:
            return RPI + 0.03
    
    def repayment_threshold(self) -> int:
        return 28470
    
    def repayment_rate(self) -> float:
        return 0.09
    
    def loan_writeoff(self) -> int:
        return 30
    
    def forecast_plot(self):
        """Enhanced plot showing interest rate variation for Plan 2."""
        df = self.forecast_to_dataframe(monthly=False)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10)) #type: ignore
        
        # Top plot: Balance and repayments
        ax1.plot(df.index, df['balance'], linewidth=2, label='Loan Balance', marker='o')
        ax1.plot(df.index, df['repayment'], linewidth=2, label='Annual Repayments', marker='s')
        ax1.set_ylabel('Amount (£)', fontweight='bold')
        ax1.set_title('Forecast of Student Loan Balance', fontweight='bold', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.get_xaxis().get_major_formatter().set_useOffset(False) 
        # Bottom plot: Interest rate
        ax2.plot(df.index, df['interest_rate'] * 100, linewidth=2, 
                color='red', marker='d')
        ax2.set_xlabel('Year', fontweight='bold')
        ax2.set_ylabel('Interest Rate (%)', fontweight='bold')
        ax2.set_title('Interest Rate Over Time', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        plt.style.use("classic")
        ax2.get_xaxis().get_major_formatter().set_useOffset(False) 
        plt.tight_layout()
        
        return plt.show() #type: ignore

    
    
    
class Plan5Loan(StudentLoanPlan):
    """Plan 5 loans: Started 2022 onwards."""
    
    def interest_rate(self, salary: Optional[float] = None) -> float:
        return RPI
    
    def repayment_threshold(self) -> int:
        return 25000
    
    def repayment_rate(self) -> float:
        return 0.09
    
    def loan_writeoff(self) -> int:
        return 40


if __name__ == "__main__":
    loan = Plan5Loan(
        balance=45000,
        salary=28000,
        year_end=date(2025, 9, 1)
    )
    
    # Compare forecasts
    print("===INPUTS===")
    print(f'your current balance is: £{loan.balance}')
    print(f'Your current salary is: £{loan.salary}')
    print(f'The year you started your course was {loan.year_end}')
    print(f'You are annually paying an additional: £{EARLY_REPAYMENT}')
    print("=== ANNUAL FORECAST ===")
    annual_df = loan.forecast_to_dataframe(monthly=False)
    print(annual_df.tail())
    print(f"\nFinal balance (annual): £{annual_df['balance'].iloc[-1]:,.2f}")
    
    print("\n=== MONTHLY FORECAST ===")
    monthly_df = loan.forecast_to_dataframe(monthly=True)
    print(monthly_df.tail())
    print(f"\nFinal balance (monthly): £{monthly_df['balance'].iloc[-1]:,.2f}")
    
    print("\n=== SUMMARY ===")
    print(f"Total lifetime repayment: £{loan.total_lifetime_repayment():,.2f}")
    print(f"Present value of repayments: £{loan.present_value_of_repayments():,.2f}")
    
    
    loan.forecast_plot()
    loan.monthly_forecast_plot()
    loan.present_value_plot()