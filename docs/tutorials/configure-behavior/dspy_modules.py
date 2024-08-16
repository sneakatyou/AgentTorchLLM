import dspy
import os
os.environ['DSP_CACHEBOOL'] = 'False'

class BasicQAWillToWork(dspy.Signature):
    """
    You are an individual living during the COVID-19 pandemic. You need to decide your willingness to work each month and portion of your assests you are willing to spend to meet your consumption demands, based on the current situation of NYC.
    """

    history = dspy.InputField(
        desc="may contain your decision in the previous months", format=list
    )
    question = dspy.InputField(
        desc="will contain the number of COVID cases in NYC, your age and other information about the economy and your identity, to help you decide your willingness to work and consumption demands"
    )
    answer = dspy.OutputField(
        desc="will contain single float value, between 0 and 1, representing realistic probability of your willingness to work. No other information should be there."
    )


class BasicQACovid(dspy.Signature):
    """Consider a random person with the following attributes:
    * age: {age}
    * location: {location}

    There is a novel disease. It spreads through contact. It is more dangerous to older people.
    People have the option to isolate at home or continue their usual recreational activities outside.
    Given this scenario, you must estimate the person's actions based on
        1) the information you are given,
        2) what you know about the general population with these attributes.

    "There isn't enough information" and "It is unclear" are not acceptable answers.
    Give a "Yes" or "No" answer, followed by a period. Give one sentence explaining your choice.
    """

    history = dspy.InputField(
        desc="may contain your decision in the previous months", format=list
    )
    question = dspy.InputField(
        desc="will contain the number of weeks since a disease started (if specified), the number of new cases this week, the percentage change from the past month's average, and asks if the person chooses to isolate at home. It may have other information also."
    )
    answer = dspy.OutputField(
        desc="Give a 'Yes' or 'No' answer, followed by a period. No other information should be there in the answer"
    )


class Reflect(dspy.Signature):
    """
    You are an Economic Analyst.
    """

    history = dspy.InputField(desc="may contain data on previous months", format=list)
    question = dspy.InputField(desc="may contain the question you are being asked")
    answer = dspy.OutputField(
        desc="may contain your analysis of the question asked based on the data in the history"
    )


class COT(dspy.Module):
    def __init__(self, qa):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(qa)

    def forward(self, question, history):
        prediction = self.generate_answer(question=question, history=history)
        return dspy.Prediction(answer=prediction)

class BasicQAStimulus(dspy.Signature):
    """
    You are an individual living in New York City (NYC) during the COVID-19 pandemic. 
    You need to decide your willingness to work each month based on the current situation of the number of COVID cases and the financial help from the government as 'stimulus payment' you are getting.
    Stimulus payment is given on above your regular income, to support your needs and savings.
    Give your willingness to work on a scale of 0 to 1, where: 0 represents 0% willingness to work and 1 represents 100% willingness to work.
    """
    # context = dspy.InputField(desc="may contain relevant facts")
    history = dspy.InputField(desc="may contain your decision in the previous months",format = list)
    question = dspy.InputField(desc="will contain the number of COVID cases in NYC, your age and stimulus payment per month, to help you decide your willingness to work. ")
    # answer = dspy.OutputField(desc="often between 1 and 5 words representing realistic probability between 0 and 1 representing the fraction of your monthly income you would want to spend this month")
    answer = dspy.OutputField(desc="must only contain an int value between 0 and 1 representing realistic probability of your willingness to work. ")

class BasicQAStimulusFatigue(dspy.Signature):
    """
    You are an individual living in New York City (NYC) during the COVID-19 pandemic. 
    You need to decide your willingness to work each month based on the current situation of the number of COVID cases, the financial help from the government as 'stimulus payment' you are getting and number of months it's been since covid started.
    Stimulus payment is given on above your regular income, to support your needs and savings.
    Pandemic fatigue is distress that can result in demotivation to follow the recommended protective behaviours, emerging gradually over time and being affected by a few emotions, experiences, and perceptions. It can influence your willingness to work by increasing your willingness to work as your savings might have decreased by now and you are tired of staying in.
    Give your willingness to work on a scale of 0 to 1, where: 0 represents 0% willingness to work and 1 represents 100% willingness to work.
    """
    # context = dspy.InputField(desc="may contain relevant facts")
    history = dspy.InputField(desc="may contain your decision in the previous months",format = list)
    question = dspy.InputField(desc="will contain the number of COVID cases in NYC, your age, stimulus payment per month and number of months it's been since covid started, to help you decide your willingness to work. ")
    # answer = dspy.OutputField(desc="often between 1 and 5 words representing realistic probability between 0 and 1 representing the fraction of your monthly income you would want to spend this month")
    answer = dspy.OutputField(desc="must only contain an int value between 0 and 1 representing realistic probability of your willingness to work. ")

# class BasicQAFatigue(dspy.Signature):
#     """
#     You are an individual living in New York City (NYC) during the COVID-19 pandemic. 
#     You need to decide your willingness to work each month based on the current situation of the number of COVID cases and number of months it's been since covid started.
#     Pandemic fatigue is distress that can result in demotivation to follow the recommended protective behaviours, emerging gradually over time and being affected by a few emotions, experiences, and perceptions. It can influence your willingness to work by increasing your willingness to work as your savings might have decreased by now and you are tired of staying in.
#     Give your willingness to work on a scale of 0 to 1, where: 0 represents 0% willingness to work and 1 represents 100% willingness to work.
#     """
#     # context = dspy.InputField(desc="may contain relevant facts")
#     history = dspy.InputField(desc="may contain your decision in the previous months",format = list)
#     question = dspy.InputField(desc="will contain the number of COVID cases in NYC, your age, and number of months it's been since covid started, to help you decide your willingness to work. ")
#     # answer = dspy.OutputField(desc="often between 1 and 5 words representing realistic probability between 0 and 1 representing the fraction of your monthly income you would want to spend this month")
#     answer = dspy.OutputField(desc="must only contain an int value between 0 and 1 representing realistic probability of your willingness to work. ")

class BasicQAFatigue(dspy.Signature):
    """
    You are an individual living in New York City (NYC) during the COVID-19 pandemic. 
    You need to decide your willingness to work each month based on the current situation of the number of COVID cases and number of months it's been since covid started.
    Pandemic fatigue is distress emerging gradually over time and being affected by a few emotions, experiences, and perceptions. It can increase your willingness to work as your savings might have decreased by now and you are tired of staying in.
    Give your willingness to work on a scale of 0 to 1, where: 0 represents 0% willingness to work and 1 represents 100% willingness to work.
    """
    # context = dspy.InputField(desc="may contain relevant facts")
    history = dspy.InputField(desc="may contain your decision in the previous months",format = list)
    question = dspy.InputField(desc="will contain the number of COVID cases in NYC, your age, and number of months it's been since covid started, to help you decide your willingness to work. ")
    # answer = dspy.OutputField(desc="often between 1 and 5 words representing realistic probability between 0 and 1 representing the fraction of your monthly income you would want to spend this month")
    answer = dspy.OutputField(desc="must only contain an int value between 0 and 1 representing realistic probability of your willingness to work. ")


# class BasicQA(dspy.Signature):
#     """
#     You are an individual residing in New York City (NYC) during the COVID-19 pandemic outbreak. 

#     At the beginning of each month, you must evaluate the current state of the pandemic in NYC and the financial assistance you are receiving from the government in the form of a 'stimulus payment.' This stimulus payment is provided in addition to your regular income to support your essential needs and savings during these challenging times.

#     Your task is to determine your willingness to work in order to earn income for the upcoming month, based on the following key factors:

#     1. The number of new COVID-19 cases reported in NYC for the previous month. Higher case numbers may make you more hesitant to work due to increased exposure risks.

#     2. The amount of stimulus payment you will receive for the upcoming month. A higher stimulus payment may alleviate financial pressures, potentially influencing your willingness to work from your workplace.
    
#     3. Your age. Your age may influence your risk perception.

#     You will be provided with the following information each month:
#     - Your current age
#     - The number of new COVID-19 cases in NYC for the previous month
#     - The amount of stimulus payment you will receive for the upcoming month

#     Based on this information, you need to express your willingness to work for the upcoming month on a scale of 0 to 1, in increments of 0.2, where:
#     0 represents an absolute unwillingness to work from your workplace
#     1 represents a complete willingness to work from your workplace
#     """
#     # context = dspy.InputField(desc="may contain relevant facts")
#     question = dspy.InputField(desc="will contain the number of COVID cases in NYC, your age and stimulus payment per month, to help you decide your willingness to work. ")
#     # answer = dspy.OutputField(desc="often between 1 and 5 words representing realistic probability between 0 and 1 representing the fraction of your monthly income you would want to spend this month")
#     answer = dspy.OutputField(desc="an int value between 0 and 1 representing realistic probability of your willingness to work, with 0 being not willing to work and 1 being willing to work.")

class BasicQAWithoutStimulus(dspy.Signature):
    """
    You are an individual living in New York City (NYC) during the COVID-19 pandemic. You need to decide your willingness to work each month based on the current situation of the number of COVID cases.
    """
    # context = dspy.InputField(desc="may contain relevant facts")
    history = dspy.InputField(desc="may contain your decision in the previous months",format = list)
    question = dspy.InputField(desc="will contain the number of COVID cases in NYC,  your age, to help you decide your willingness to work. ")
    answer = dspy.OutputField(desc="must only contain an int value between 0 and 1 representing realistic probability of your willingness to work")
    
class BasicQANoCases(dspy.Signature):
    """
    You are an individual living in New York City (NYC). You need to decide your willingness to work each month.
    """
    # context = dspy.InputField(desc="may contain relevant facts")
    history = dspy.InputField(desc="may contain your decision in the previous months",format = list)
    question = dspy.InputField(desc="will contain your age, to help you decide your willingness to work. ")
    answer = dspy.OutputField(desc="must only contain an int value between 0 and 1 representing realistic probability of your willingness to work")