# ============================================
# FINAL PROJECT
# Educational Chatbot (Rule + AI Based)
# ============================================

import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------------------
# Rule Based Responses
# ---------------------------

def rule_based_response(user_input):
    user_input = user_input.lower()
    
    if "admission" in user_input:
        return "Admissions are open. Please visit www.collegeadmission.com"
    
    elif "library" in user_input:
        return "Library timing: 9 AM to 6 PM. Over 20,000 books available."
    
    elif "exam" in user_input:
        return "Exam starts from 10th March. Hall ticket available online."
    
    elif "fee" in user_input:
        return "You can pay fees online through student portal."
    
    elif "faq" in user_input:
        return "You can ask about admission, library, exams or fees."
    
    return None

# ---------------------------
# AI Based Model
# ---------------------------

training_sentences = [
    "How to apply admission",
    "Library timing",
    "Exam date",
    "How to pay fees",
    "When are exams",
    "Books available in library"
]

training_labels = [
    "Admission process details available on website.",
    "Library open from 9 AM to 6 PM.",
    "Exams start from 10th March.",
    "Fees can be paid via student portal.",
    "Exams start from 10th March.",
    "Library has 20,000+ books."
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

model = MultinomialNB()
model.fit(X, training_labels)

def ai_response(user_input):
    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)
    return prediction[0]

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("🎓 Educational Chatbot System")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Ask your question:")

if user_input:
    
    # Rule Based First
    response = rule_based_response(user_input)
    
    # If rule not found → AI
    if response is None:
        response = ai_response(user_input)
    
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", response))

# Display Chat
for sender, message in st.session_state.chat:
    if sender == "You":
        st.write("🧑:", message)
    else:
        st.write("🤖:", message)
