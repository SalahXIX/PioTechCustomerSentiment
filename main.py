import streamlit
from langchain_community.llms import ollama
import json
import os

intent_list = ["Compliment", "Feedback", "Appreciation", "Request for Information", "Inquiry", 
               "Issue Report", "Confirmation", "Access Request", "Administrative Request", "Complaint", 
               "SLA Dispute", "Escalation Request", "Refund Request", "Suggestion / Feature Request", "Clarification", "Follow-up", 
               "Acknowledgment", "Dispute", "Caution / Process Improvement", "Business Request"]
sentiment_list = ["Positive", "Neutral", "Mixed feelings", "Negative", "Confused"]

model = ollama.Ollama(model="llama3")


Sentiment_Template= '''
You are a Sentiment Analyzer AI, you analyze the text given to you and figure out its sentiment/tone.
You choose the sentiment that most closely matches the sentiment of the text you are provided based on the following options: {sentiment_list}.
The text: {text}
Your answer (only choose from the provided list):
'''


Intent_Template= '''
You are an intent Analyzer AI, you analyze the text given to you and figure out what the reason it was sent.
You choose the intent that most closely matches the intent of the text you are provided based on the following options: {intent_list}.
The text: {text}
Your answer (only choose from the provided list):
'''

def first_interperter(Answer):
    for option in sentiment_list:
        if option.lower() in str(Answer).strip().lower():
            return option
    return "Unclear"    

def second_interperter(Answer):
    for option in intent_list:
        if option.lower() in str(Answer).strip().lower():
            return option
    return "Unclear"    

        

def Read_Texts(MarkdownPath):
    TextList = []
    with open(MarkdownPath, 'r') as f:
        TextList.extend(line.strip() for line in f if line.strip())    
    return TextList    

def Evaluate_Texts(TextList):
    results=[]
    for text in TextList:
        results.append(text)
        Sentiment_Prompt = Sentiment_Template.format(sentiment_list=sentiment_list, text=text)
        sentiment = first_interperter(model.invoke(Sentiment_Prompt))
        Intent_Prompt = Intent_Template.format(intent_list=intent_list, text=text)
        intent = second_interperter(model.invoke(Intent_Prompt))
        entry = {
            "Email_Text": text,
            "Sentiment": sentiment,
            "Intent": intent
        }
        save_responses([entry])
        results.append(entry)
    return results

def save_responses(responses, filename="CustomerSentiment.json"):
    existing = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                existing = json.load(f)
            except:
                pass
    existing.extend(responses)
    with open(filename, "w") as f:
        json.dump(existing, f, indent=2)

textlist = Read_Texts("/Users/salahalalix/Desktop/pio-tech/Sentiment_Analyzer/Examples.md")
result = Evaluate_Texts(textlist)

        
