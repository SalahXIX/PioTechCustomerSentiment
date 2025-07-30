import streamlit as st
import json
import os
from transformers import pipeline


intent_list = ["Compliment", "Feedback", "Appreciation", "Request for Information", "Inquiry", 
               "Issue Report", "Confirmation", "Access Request", "Administrative Request", "Complaint", 
               "SLA Dispute", "Escalation Request", "Refund Request", "Suggestion / Feature Request", "Clarification", "Follow-up", 
               "Acknowledgment", "Dispute", "Caution / Process Improvement", "Business Request"]

intent_list = sorted(intent_list, key=lambda x: -len(x))


sentiment_list = ["Positive", "Neutral", "Mixed feelings", "Negative", "Confused"]


sentiment_model = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

def invoke_model(prompt):
    output = sentiment_model(prompt)[0]["generated_text"]
    return output.replace(prompt, "").strip()

Sentiment_Template= '''
You are a Sentiment Analyzer AI, you analyze the text given to you and figure out its sentiment/tone.
You choose the sentiment that most closely matches the sentiment of the text you are provided based on the following options: {sentiment_list}.
The text: {text}
Your answer (only choose from the provided list):
'''


Intent_Template = '''
You are an intent classification AI. You must classify the intent of the following text into one of these labels:
{intent_list}.
Only output the exact intent label — nothing else.
Text: "{text}"
Intent:
'''

def first_interperter(Answer):
    for option in sentiment_list:
        if option.lower() in str(Answer).strip().lower():
            return option
    return "Unclear"    


def second_interperter(Answer):
    Answer = str(Answer).strip().lower()
    for intent in intent_list:
        if Answer == intent.lower():
            return intent
    for intent in intent_list:
        if intent.lower() in Answer:
            return intent
    for intent in intent_list:
        keywords = intent.lower().replace("/", "").split()
        if any(word in Answer for word in keywords):
            return intent
    return "Unclear"

        

def Read_Texts(MarkdownPath):
    TextList = []
    with open(MarkdownPath, 'r') as f:
        TextList.extend(line.strip() for line in f if line.strip())    
    return TextList    

def Evaluate_Texts(TextList):
    results = []
    for text in TextList:

        Sentiment_Prompt = Sentiment_Template.format(sentiment_list=sentiment_list, text=text)
        sentiment = first_interperter(invoke_model(Sentiment_Prompt))

        classification = intent_classifier(text, intent_list, multi_label=False)
        top_intent = classification["labels"][0]
        intent = second_interperter(top_intent)

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


st.set_page_config(page_title="Customer Sentiment Analyzer", layout="centered")

st.markdown("<h1 style='text-align: center;'>📊 Customer Sentiment & Intent Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze customer feedback/Emails for sentiment and purpose using AI</p>", unsafe_allow_html=True)


input_mode = st.radio("Choose input method:", ["Manual Input", "Upload File"])

textlist = []

if input_mode == "Manual Input":
    user_input = st.text_area("Enter one or more sentences (one per line):")
    if user_input:
        textlist = [line.strip() for line in user_input.splitlines() if line.strip()]
else:
    uploaded_file = st.file_uploader("Upload a `.txt` or `.md` file", type=["txt", "md"])
    if uploaded_file is not None:
        textlist = Read_Texts(uploaded_file)

# Run classification
if st.button("Classify Text"):
    if not textlist:
        st.warning("Please enter text or upload a file before classifying.")
    else:
        with st.spinner("Analyzing..."):
            results = Evaluate_Texts(textlist)
        st.success("Analysis complete!")

        for entry in results:
            st.markdown("---")
            st.markdown(f"**Text:** {entry['Email_Text']}")
            st.markdown(f"**Sentiment:** `{entry['Sentiment']}`")
            st.markdown(f"**Intent:** `{entry['Intent']}`")

        
