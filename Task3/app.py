import streamlit as st
import spacy
from transformers import pipeline, Conversation
import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", "Some weights of the model checkpoint at")

# --- 1. MODEL LOADING (Cached) ---
@st.cache_resource
def load_models():
    print("Loading models... This will run only once.")
    nlp = spacy.load("en_core_web_sm")
    chatbot = pipeline("conversational", model="facebook/blenderbot-400M-distill")
    return nlp, chatbot

nlp, chatbot = load_models()

# --- 2. CHATBOT LOGIC FUNCTION (No changes here) ---
def get_bot_response(user_input, nlp_model, hf_chatbot, hf_conversation):
    """
    Gets a response from the chatbot (spaCy router or HF fallback).
    """
    
    # --- Step 3: spaCy "Smart Router" Logic ---
    doc = nlp_model(user_input)
    user_input_lower = user_input.lower().strip()

    # RULE 1: "About" intent
    if "who are you" in user_input_lower or "about you" in user_input_lower:
        return "I am a hybrid chatbot built with spaCy and Hugging Face!"
    
    # RULE 2: "Time" intent
    if "what time" in user_input_lower or "current time" in user_input_lower:
        found_location = False
        for ent in doc.ents:
            if ent.label_ == "GPE":
                found_location = True
                break
        if found_location:
            return "I'm sorry, I can only tell you my local time. I don't have access to other timezones."
        else:
            now = datetime.datetime.now()
            return f"The current (local) time is {now.strftime('%I:%M %p')}."

    # RULE 3a: Block specific keywords.
    fact_keywords = ["review", "rating", "plot", "summary", "release date"]
    if any(keyword in user_input_lower for keyword in fact_keywords):
        return "I'm a conversational bot and don't have access to facts, movie reviews, or the internet."
    
    # RULE 3b: Block questions about a specific PERSON.
    has_person = any(ent.label_ == "PERSON" for ent in doc.ents)
    if has_person and ("who" in user_input_lower or "?" in user_input_lower):
        return "I'm sorry, I don't have access to specific information about people."

    # RULE 4: General "Location" mention
    found_gpe_general = False
    for ent in doc.ents:
        if ent.label_ == "GPE" and "time" not in user_input_lower and not has_person:
            return f"I don't have specific data on {ent.text}, but it sounds like a nice place."

    # --- Step 4: Hugging Face "Generative" Fallback ---
    hf_conversation.add_user_input(user_input)
    hf_chatbot(hf_conversation)
    return hf_conversation.generated_responses[-1]

# --- 3. STREAMLIT APP UI (No changes here) ---
st.title("🤖 My Hybrid AI Chatbot")
st.caption("Using spaCy for rules and Hugging Face for conversation.")

st.sidebar.title("--- My Limitations ---")
st.sidebar.info(
    """
    1. I am a language model, not a search engine. I don't know facts, news, or reviews.
    2. My knowledge is frozen in time (around 2020).
    3. I can only tell you my local time, not for other timezones.
    4. I may forget what we talked about in a very long conversation.
    5. Please ask one thing at a time.
    """
)

# --- 4. SESSION STATE INITIALIZATION (No changes here) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your hybrid chatbot. How can I help?"}
    ]
if "conversation" not in st.session_state:
    st.session_state.conversation = Conversation()

# --- 5. DISPLAY CHAT HISTORY (No changes here) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. CHAT INPUT AND RESPONSE LOGIC (FIX ADDED) ---
if prompt := st.chat_input("Ask me anything..."):
    
    # 1. Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- THIS IS THE FIX ---
    # Get the current conversation
    current_conversation = st.session_state.conversation
    
    # Check history length. Let's keep it to 3 exchanges (6 total messages).
    # We check *past_user_inputs* since the bot hasn't replied yet.
    if len(current_conversation.past_user_inputs) > 3:
        # Tell the user we are resetting memory
        reset_message = "My short-term memory is full! I'm starting our chat over."
        st.session_state.messages.append({"role": "assistant", "content": reset_message})
        with st.chat_message("assistant"):
            st.info(reset_message) # st.info gives it a blue box
        
        # Create a brand new, empty conversation
        st.session_state.conversation = Conversation()
    # --- END FIX ---
        
    # 2. Get bot response by calling our main logic function
    response = get_bot_response(
        prompt,
        nlp,
        chatbot,
        st.session_state.conversation # Pass the (possibly new) conversation object
    )
    
    # 3. Add bot response to session state and display it
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)