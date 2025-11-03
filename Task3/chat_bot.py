import spacy
from transformers import pipeline, Conversation
import warnings
import datetime 

warnings.filterwarnings("ignore", "Some weights of the model checkpoint at")

# --- Step 1: Load Models ---
print("Bot: Loading spaCy 'en_core_web_sm' model...")
nlp = spacy.load("en_core_web_sm")

print("Bot: Loading Hugging Face 'facebook/blenderbot-400M-distill' model...")
chatbot = pipeline("conversational", model="facebook/blenderbot-400M-distill")

# CHANGED 2: Initialize the correct object, not a list
conversation_history = Conversation() 

# --- Welcome Message (No changes) ---
def print_welcome_message():
    print("--------------------------------------------------")
    print("Bot: All models loaded. Chatbot is ready!")
    print("Bot: I am a hybrid conversational bot.")
    print("\n--- My Limitations ---")
    print("1. I am a language model, not a search engine. I don't know facts, news, or reviews.")
    print("2. My knowledge is frozen in time (around 2020).")
    print("3. I can only tell you my local time, not for other timezones.")
    print("4. I may forget what we talked about in a very long conversation.")
    print("5. Please ask one thing at a time.")
    print("--------------------------------------------------")
    print("Bot: Type 'quit' or 'exit' to end the conversation.")
    print("--------------------------------------------------")

# --- Step 2: Main Chat Loop ---
def start_chat():
    global conversation_history  # Use the global Conversation object
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye!")
            break

        # --- Step 3: spaCy "Smart Router" Logic (No changes) ---
        doc = nlp(user_input) 
        user_input_lower = user_input.lower().strip()

        # RULE 1: "About" intent
        if "who are you" in user_input_lower or "about you" in user_input_lower:
            print("Bot: I am a hybrid chatbot built with spaCy and Hugging Face!")
            continue 
            
        # RULE 2: "Time" intent
        if "what time" in user_input_lower or "current time" in user_input_lower:
            found_location = False
            for ent in doc.ents:
                if ent.label_ == "GPE":
                    found_location = True
                    break
            if found_location:
                print("Bot: I'm sorry, I can only tell you my local time. I don't have access to other timezones.")
            else:
                now = datetime.datetime.now()
                print(f"Bot: The current (local) time is {now.strftime('%I:%M %p')}.")
            continue 

        # RULE 3a: Block specific keywords.
        fact_keywords = ["review", "rating", "plot", "summary", "release date"]
        if any(keyword in user_input_lower for keyword in fact_keywords):
            print("Bot: I'm a conversational bot and don't have access to facts, movie reviews, or the internet.")
            continue
            
        # RULE 3b: Block questions about a specific PERSON.
        has_person = any(ent.label_ == "PERSON" for ent in doc.ents)
        if has_person and ("who" in user_input_lower or "?" in user_input_lower):
            print("Bot: I'm sorry, I don't have access to specific information about people.")
            continue

        # RULE 4: General "Location" mention
        found_gpe_general = False
        for ent in doc.ents:
            if ent.label_ == "GPE" and "time" not in user_input_lower and not has_person:
                print(f"Bot: I don't have specific data on {ent.text}, but it sounds like a nice place.")
                found_gpe_general = True
                break
        
        if found_gpe_general:
            continue 

        # --- Step 4: Hugging Face "Generative" Fallback (UPDATED) ---
        
        # CHANGED 3: Use .add_user_input()
        conversation_history.add_user_input(user_input)
        
        # The pipeline takes the Conversation object and updates it
        chatbot(conversation_history)
        
        # CHANGED 4: Get the response from .generated_responses
        print("Bot:", conversation_history.generated_responses[-1])

if __name__ == "__main__":
    print_welcome_message() 
    start_chat()