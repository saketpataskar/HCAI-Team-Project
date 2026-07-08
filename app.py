import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline
from groq import Groq
from system_instruction import system_instructions

st.set_page_config(page_title="AI Python Tutor", page_icon="🤖", layout="centered")
st.title("🤖 AI Python Tutor")
st.caption("A safe, empathetic space to learn programming.")

@st.cache_resource
def load_app():
    
    # Loading Groq
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # Loading sentiment analysis model
    sentiment_classifier = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base"
    )
    
    # Loading vector DB
    db_client = chromadb.PersistentClient(path="./tutor_dbV1")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    db_collection = db_client.get_collection(
        name="python_textbook", 
        embedding_function=sentence_transformer_ef
    )
    
    return groq_client, sentiment_classifier, db_collection

client, sentiment_classifier, db_collection = load_app()

def extract_search_topic(user_input):
    # Uses the LLM to find the textbook topic without noise.
    prompt = f"""Extract the core Python programming concept from this student's message. 
    Respond ONLY with the technical keyword(s) to search in a textbook. Do not write a sentence.
    If there is no clear topic, respond with 'Python basics'.
    
    Student message: "{user_input}"
    Keyword(s):"""
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except:
        return user_input
    
def get_textbook_context(query_text):
    try:
        res = db_collection.query(query_texts=[query_text], n_results=2)
        if res['documents'] and res['documents'][0]:
            return "\n\n".join(res['documents'][0])
        return ""
    except Exception:
        return ""
    
def detect_anxiety(text):
    prediction = sentiment_classifier(text)[0]
    emotion = prediction['label']
    triggers = ['fear', 'sadness', 'anger', 'disgust']
    if emotion in triggers:
        return f"High (Student is feeling {emotion})"
    return f"Normal (Student is feeling {emotion})"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["display_content"])


if user_query := st.chat_input("Ask a question about Python..."):
    with st.chat_message("user"):
        st.markdown(user_query)

    anx = detect_anxiety(user_query)
    topic = extract_search_topic(user_query)

    st.caption("Thinking....")
    context_from_text = get_textbook_context(topic)

    prom = f"Student input: '{user_query}'\nEmotional state: {anx}\nTextbook Context:\n{context_from_text}"
    st.session_state.chat_history.append({"role": "user", "display_content": user_query, "ai_content": prom})

    sys_ins = system_instructions
    messages_for_api = [{"role": "system", "content": system_instructions}]
    for msg in st.session_state.chat_history:
        messages_for_api.append({"role": msg["role"], "content": msg.get("ai_content", msg["display_content"])})

    with st.chat_message("Tutor"):
        with st.spinner("Generating Answers..."):
            try: 
                response = client.chat.completions.create(
                    messages=messages_for_api,
                    model="llama-3.3-70b-versatile",
                )
                tutor_res = response.choices[0].message.content
                st.markdown(tutor_res)
                
                # F. Save AI reply to memory
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "display_content": tutor_res,
                    "ai_content": tutor_res
                })
            except Exception as e:
                st.error(f"Error connecting: {e}")
