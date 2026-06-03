# full_app_code_no_keys.py
import sys
import os # <<< os اضافه شد
import json
import base64
from openai import OpenAI
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from typing import List

import auth

# --- لاگ اولیه برای اطمینان از اجرای کد ---
print(">>> LOG: app.py execution started. Initializing...")

auth.init_db()

# --- Helper Function for Image ---
def get_image_base64(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    return base64.b64encode(bytes_data).decode('utf-8')

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
    
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

if not st.session_state.logged_in:
    auth.show_login_page()
else:
    # --- Sidebar & Chat History Management ---
    st.sidebar.write(f"👤 User: **{st.session_state.username}**")
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.session_state.current_chat_id = None
        st.rerun()

    st.sidebar.divider()
    
    if st.sidebar.button("➕ New Chat", use_container_width=True):
        new_id = auth.create_new_chat(st.session_state.username)
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.sidebar.write("**Active Chats:**")
    active_chats = auth.get_user_chats(st.session_state.username)
    archived_chats = auth.get_archived_chats(st.session_state.username)
    
    all_valid_chat_ids = [c[0] for c in active_chats] + [c[0] for c in archived_chats]
    if st.session_state.current_chat_id not in all_valid_chat_ids:
        if active_chats:
            st.session_state.current_chat_id = active_chats[0][0]
        else:
            default_chat_id = auth.create_new_chat(st.session_state.username)
            st.session_state.current_chat_id = default_chat_id
            active_chats = auth.get_user_chats(st.session_state.username)

    for chat_id, title in active_chats:
        col1, col2 = st.sidebar.columns([8, 2])
        is_active = (chat_id == st.session_state.current_chat_id)
        btn_label = f"💬 {title}" if is_active else title
        
        with col1:
            if st.button(btn_label, key=f"chat_btn_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()
                
        with col2:
            with st.popover("⋮", use_container_width=True):
                new_title = st.text_input("Rename", value=title, key=f"ren_{chat_id}")
                if st.button("Save", key=f"save_{chat_id}"):
                    auth.update_chat_title(chat_id, new_title)
                    st.rerun()
                
                if st.button("Archive 📁", key=f"arc_{chat_id}"):
                    auth.archive_chat(chat_id)
                    st.rerun()
                
                st.divider()
                st.markdown("**Delete Chat**")
                confirm_delete = st.checkbox("Are you sure?", key=f"chk_del_{chat_id}")
                if confirm_delete:
                    if st.button("Confirm Delete 🗑️", key=f"del_{chat_id}"):
                        auth.delete_chat(chat_id)
                        st.rerun()

    if archived_chats:
        with st.sidebar.expander("📁 Archived Chats"):
            for chat_id, title in archived_chats:
                col1, col2 = st.columns([8, 2])
                is_active = (chat_id == st.session_state.current_chat_id)
                btn_label = f"📄 {title}" if is_active else title
                
                with col1:
                    if st.button(btn_label, key=f"arch_btn_{chat_id}", use_container_width=True):
                        st.session_state.current_chat_id = chat_id
                        st.rerun()
                with col2:
                    with st.popover("⋮", key=f"arch_pop_{chat_id}"):
                        st.markdown("**Delete Chat**")
                        confirm_delete = st.checkbox("Are you sure?", key=f"chk_del_arch_{chat_id}")
                        if confirm_delete:
                            if st.button("Confirm Delete 🗑️", key=f"del_arch_{chat_id}"):
                                auth.delete_chat(chat_id)
                                st.rerun()
    
    # --- Core Application Configuration ---
    # کلیدها از اینجا حذف شدند و از st.secrets یا os.environ خوانده می شوند
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        BASE_URL = st.secrets["BASE_URL"]
        print(">>> LOG: Successfully loaded secrets from st.secrets.")
    except KeyError:
        st.error("FATAL ERROR: `OPENAI_API_KEY` or `BASE_URL` not found in Streamlit Secrets. Please set them.")
        print(">>> ERROR: Secrets not found. Stopping execution.")
        st.stop()
    

    st.title("✈️ Aviation AI Assistant")

    @st.cache_resource
    def load_vector_db():
        print(">>> LOG: Creating OpenAIEmbeddings object...")
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY, 
            openai_api_base=BASE_URL,
            request_timeout=30
        )
        print(">>> LOG: Loading FAISS database...")
        db = FAISS.load_local(
            "my_vector_db", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(">>> LOG: FAISS database loaded successfully.")
        return db.as_retriever(search_kwargs={"k": 3})

    try:
        print(">>> LOG: Attempting to load vector database...")
        retriever = load_vector_db()
        print(">>> LOG: Vector database loaded and retriever created.")
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        print(f">>> ERROR: Failed to load vector DB: {e}")
        st.stop()

    print(">>> LOG: Initializing ChatOpenAI and OpenAI client...")
    llm = ChatOpenAI(
        model_name="gpt-4o", 
        openai_api_key=OPENAI_API_KEY, 
        base_url=BASE_URL,
        request_timeout=45
    )
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY, 
        base_url=BASE_URL,
        timeout=30
    )
    print(">>> LOG: LLM and OpenAI clients initialized successfully.")
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    
    db_messages = auth.get_chat_messages(st.session_state.current_chat_id)
    
    for i in range(0, len(db_messages) - 1, 2):
        if db_messages[i]["role"] == "user" and i+1 < len(db_messages):
            memory.save_context({"question": db_messages[i]["content"]}, 
                                {"answer": db_messages[i+1]["content"]})

    for msg in db_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and len(msg.get("sources", [])) > 0:
                with st.expander("📑 Sources Used"):
                    for i, doc_dict in enumerate(msg["sources"]):
                        page_number = doc_dict.get('page', 0) + 1
                        st.write(f"**Source {i+1} (Page {page_number}):**")
                        st.write(doc_dict.get('content', ''))

    is_current_archived = any(c[0] == st.session_state.current_chat_id for c in archived_chats)
    
    if is_current_archived:
        st.warning("این چت آرشیو شده است و فقط قابل خواندن میباشد.")
    else:
        col1, col2, _ = st.columns([2, 2, 6])
        
        with col1:
            with st.popover("Attach", icon=":material/attachment:", help="Upload Image or Map"):
                uploaded_image = st.file_uploader(
                    "Select File", 
                    type=["png", "jpg", "jpeg"], 
                    key=f"img_uploader_{st.session_state.current_chat_id}_{st.session_state.file_uploader_key}"
                )
                    
        with col2:
            with st.popover("Voice", icon=":material/mic:", help="Record a voice message"):
                recorded_audio = st.audio_input(
                    "Record your message",
                    key=f"audio_uploader_{st.session_state.current_chat_id}_{st.session_state.file_uploader_key}"
                )
            
        if uploaded_image:
            st.info(f"✅ Image '{uploaded_image.name}' attached.")
        if recorded_audio:
            st.info("✅ Voice message recorded. Submit to transcribe and ask.")

        text_prompt = st.chat_input("Ask your question...")
        final_prompt = None
        
        if text_prompt:
            final_prompt = text_prompt
        elif recorded_audio:
            with st.spinner("🗣️ Transcribing voice..."):
                try:
                    print(">>> LOG: Sending audio to Whisper API...")
                    recorded_audio.name = "audio.wav" 
                    transcript = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=recorded_audio
                    )
                    print(">>> LOG: Audio transcription successful.")
                    final_prompt = transcript.text
                except Exception as e:
                    print(f">>> ERROR: Audio transcription failed: {e}")
                    st.error(f"خطا در تبدیل صدا به متن. لطفاً دوباره تلاش کنید: {e}")
                    st.stop()

        if final_prompt:
            if len(db_messages) == 0:
                auth.update_chat_title(st.session_state.current_chat_id, final_prompt[:30] + "...")
            
            with st.chat_message("user"): 
                st.markdown(final_prompt)
                if uploaded_image:
                    st.image(uploaded_image, width=300)
                    
            auth.save_message(st.session_state.current_chat_id, "user", final_prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking & Analyzing..."):
                    try:
                        print(">>> LOG: Generating embeddings and searching FAISS...")
                        source_docs = retriever.invoke(final_prompt)
                        print(">>> LOG: FAISS search successful.")
                    except Exception as e:
                        print(f">>> ERROR: Retriever (Embeddings) failed: {e}")
                        st.error(f"ارتباط با سرور برای جستجوی داکیومنت قطع شد (Embeddings Error): {e}")
                        st.stop()

                    context_text = "\n\n".join([doc.page_content for doc in source_docs])
                    
                    system_prompt = f"""Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
IMPORTANT INSTRUCTIONS:
1. You MUST answer entirely in ENGLISH.
2. You MUST cite the source page numbers in your answer (e.g., "According to Page 5...").
3. If an image or map is provided, analyze it carefully along with the text context to give a precise answer.
Context:
{context_text}
Question: {final_prompt}
Helpful Answer in English:"""
                    
                    message_content = [{"type": "text", "text": system_prompt}]
                    
                    if uploaded_image:
                        base64_image = get_image_base64(uploaded_image)
                        message_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        })
                    
                    human_msg = HumanMessage(content=message_content)
                    chat_history = memory.chat_memory.messages
                    
                    try:
                        print(">>> LOG: Sending prompt and context to LLM (gpt-4o)...")
                        result = llm.invoke(chat_history + [human_msg])
                        print(">>> LOG: LLM response received successfully.")
                        answer = result.content
                    except Exception as e:
                        print(f">>> ERROR: LLM invocation failed: {e}")
                        st.error(f"سرور پاسخ نمیدهد (LLM API Error). ممکن است اینترنت یا پراکسی مشکل داشته باشد: {e}")
                        st.stop()
                    
                    st.markdown(answer)
                    
                    sources_for_db = []
                    if len(source_docs) > 0:
                        with st.expander("📑 Extracted Document Chunks"):
                            for i, doc in enumerate(source_docs):
                                page = doc.metadata.get('page', 0)
                                content = doc.page_content
                                sources_for_db.append({"page": page, "content": content})
                                st.write(f"**Source {i+1}:**")
                                st.write(content)

                    auth.save_message(st.session_state.current_chat_id, "assistant", answer, sources_for_db)
                    memory.save_context({"question": final_prompt}, {"answer": answer})
            
            st.session_state.file_uploader_key += 1
            st.rerun()

print(">>> LOG: app.py execution finished.")

