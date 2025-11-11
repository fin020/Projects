from abc import ABC, abstractmethod
from datetime import date, datetime
from matplotlib import pyplot as plt
from typing import Optional


#interest rates:
RPI = 0.032
salary_increse_rate = 0.03
early_repayment = 0


class StudentLoanPlan(ABC):
    def __init__(self, balance:float, salary:float, year_start:date):
        self.balance = balance
        self.salary = salary
        self.year_start = year_start
            
    def get_plan(self):
        if self.year_start.year < 2012:
            return Plan1Loan(self.balance, self.salary, self.year_start)
        elif 2012<= self.year_start.year < 2022:
            return Plan2Loan(self.balance, self.salary, self.year_start)
        else:
            return Plan5Loan(self.balance, self.salary, self.year_start)
        
             
    @abstractmethod
    def interest_rate(self, salary:Optional[float]) -> float:
        """return interest rate based on current RPI conditions and potential inflation"""
        
    @abstractmethod
    def repayment_threshold(self) -> int: 
        """returns salary threshold for repayment"""
        pass
    @abstractmethod
    def repayment_rate(self)-> float:
        """returns the % of salary above the threshold that goes towards repayment"""
        pass
    @abstractmethod
    def loan_writeoff(self)-> int:
        """years the loan is written off"""
        pass
    
    def annual_repayment(self,salary:float | None = None, balance: float | None = None) -> float:
        if salary is None:
            salary = self.salary
        if balance is None:
            balance = self.balance
        threshold = self.repayment_threshold()
        rate = self.repayment_rate()
        result:float = (salary - threshold) * rate
        if balance < 0.01:
            return 0.0
        else:
            return max(0.0, result)
        
    def loan_interest(self, balance:float | None = None, salary:float | None= None) ->float:
        if balance is None:
            balance = self.balance
        if salary is None:
            salary = self.salary
        return balance * (self.interest_rate(salary))
    
    def forecast(self):
        year:int = datetime.now().year
        years_to_simulate = range(1,self.loan_writeoff() - (year - self.year_start.year))
        balance =  self.balance
        salary = self.salary
        early = early_repayment
        forecast:list[dict[int,tuple[float,float,float,float]]] = [{year: (self.balance, self.annual_repayment() ,self.salary, self.interest_rate(salary))}]
        for i in years_to_simulate:
            interest_rate = self.interest_rate(salary)
            interest = self.loan_interest(balance)
            repayment = self.annual_repayment(salary, balance)
            balance = max(0, balance + interest - repayment- early)
            if repayment + early > balance:
                repayment = balance 
            salary *= (1 + salary_increse_rate)
            forecast.append({
            year + i: (balance, repayment ,salary, interest_rate)})
        return forecast
            
    def forecast_plot(self):
        forecast_data = self.forecast()
        fig, axs= plt.subplots(2) #type:ignore
        years = [list(d.keys())[0] for d in forecast_data]           # [2020, 2021, 2022]
        balances = [list(d.values())[0][0] for d in forecast_data] 
        repayments = [list(d.values())[0][1] for d in forecast_data]# extract balance (second element)
        interest = [list(d.values())[0][3] for d in forecast_data]
        axs[0].plot(years, balances, linewidth=2, label="Loan Balance") #type:ignore
        axs[0].plot(years, repayments,linewidth=2, label="Annual Repayments") #type: ignore
        axs[1].plot(years, interest, linewidth=2) #type: ignore 
        axs[0].legend(loc="upper right")
        for ax in axs.flat:
            ax.set_xlabel("Year") #type:ignore
            ax.grid(True)
        axs[0].set_ylabel("Loan Balance (£)", fontweight="bold") #type:ignore
        axs[1].set_ylabel("Interest rate on Loan %", fontweight="bold")
        axs[1].set_title("Interest Rate over time",fontweight="bold")
        axs[0].set_title("Forecast of Student Loan Balance", fontweight="bold") #type:ignore
        plt.style.use("seaborn-v0_8-paper")
        fig.tight_layout()
        fig.set_figheight(15)
        return plt.show() #type:ignore


class Plan1Loan(StudentLoanPlan):
    def interest_rate(self, salary: Optional[float] = None) -> float:
        return RPI 
    
    def repayment_threshold(self) -> int:
        return 26065
    
    def repayment_rate(self):
        return 0.09
    
    def loan_writeoff(self):
        return 25
    

class Plan2Loan(StudentLoanPlan):
    def interest_rate(self, salary:float | None = None) -> float:
        if salary is None:
            salary = self.salary
        if salary < 28470:
            return RPI
        elif 28470 <= salary < 51245:
            differential = (salary - 28470) / 22775
            return RPI +  differential*0.03
        else: 
            return RPI + 0.03
    
    def repayment_threshold(self) -> int:
        return 28470
    
    def repayment_rate(self):
        return 0.09
    
    def loan_writeoff(self):
        return 30
    
    
    
    
class Plan5Loan(StudentLoanPlan):
    def interest_rate(self, salary: Optional[float] = None) -> float:
        return RPI 
    
    def repayment_threshold(self) -> int:
        return 25000
    
    def repayment_rate(self):
        return 0.09
    
    def loan_writeoff(self):
        return 40
        

loan = Plan2Loan(
    balance=53039,       # starting loan balance
    salary=40000,        # starting salary
    year_start=date(2018, 1, 1)  # start year
)

# Run forecast
forecast_data = loan.forecast()
print("Forecast data (year: (balance, salary)):")
for entry in forecast_data:
    print(entry)

# Plot the forecast
loan.forecast_plot()
