import sys
import json
import base64
from openai import OpenAI # <-- اضافه شده برای استفاده از Whisper

# --- Start of internet bypass for tiktoken ---
class FakeEncoding:
    def encode(self, text, *args, **kwargs): return [1] * (len(text) // 4 + 1)
    def encode_ordinary(self, text, *args, **kwargs): return [1] * (len(text) // 4 + 1)
    def decode(self, tokens, *args, **kwargs): return ""

class FakeTiktoken:
    def encoding_for_model(self, *args, **kwargs): return FakeEncoding()
    def get_encoding(self, *args, **kwargs): return FakeEncoding()

sys.modules['tiktoken'] = FakeTiktoken()
# --- End of internet bypass ---

import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import Field
from typing import List

import auth

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

    # --- Standard Retriever ---
    class SimpleKeywordRetriever(BaseRetriever):
        documents: List[Document] = Field(default_factory=list)
        k: int = 3 

        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            query_words = set(query.lower().split())
            if not query_words:
                return []
                
            scored_docs = []
            for doc in self.documents:
                text_words = doc.page_content.lower().split()
                score = sum(1 for word in query_words if word in text_words)
                if score > 0: 
                    scored_docs.append((score, doc))
                    
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return [doc for score, doc in scored_docs[:self.k]]

    PDF_FOLDER_PATH = r"docs"
    OPENAI_API_KEY = "sk-jOn337n0y1yYP7kWTQVFRzuCsvfXlA5Y56kUDkRaQeoqhORC" 
    BASE_URL = "https://api.gapgpt.app/v1" 

    st.title("✈️ Aviation AI Assistant")

    @st.cache_resource
    def process_pdf():
        loader = PyPDFDirectoryLoader(PDF_FOLDER_PATH)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        for chunk in chunks:
            source_file = chunk.metadata.get('source', 'Unknown File').split('\\')[-1]
            page_num = chunk.metadata.get('page', 0) + 1
            chunk.page_content = f"[Source File: {source_file}, Page: {page_num}]\n{chunk.page_content}"
            
        retriever = SimpleKeywordRetriever(documents=chunks, k=3)
        return retriever

    retriever = process_pdf()
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY, base_url=BASE_URL)
    
    # کلاینت خام OpenAI برای استفاده از Whisper (تبدیل صدا به متن)
    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
    
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

    # --- New Chat Input, Image Uploader & Voice Recorder ---
    is_current_archived = any(c[0] == st.session_state.current_chat_id for c in archived_chats)
    
    if is_current_archived:
        st.warning("این چت آرشیو شده است و فقط قابل خواندن می‌باشد.")
    else:
        # ایجاد دو ستون برای قرارگیری دکمه‌های امکانات جانبی
        col1, col2, _ = st.columns([2, 2, 6])
        
        with col1:
            # استفاده از آیکون متریال و نام مینیمال همراه با راهنمای شناور (help)
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

        # دریافت ورودی متنی (در صورتی که کاربر تایپ کند)
        text_prompt = st.chat_input("Ask your question...")
        
        # تعیین پرامپت نهایی
        final_prompt = None
        
        if text_prompt:
            final_prompt = text_prompt
        elif recorded_audio:
            # اگر کاربر چیزی تایپ نکرد اما صدا ضبط کرد، صدا را به متن تبدیل می‌کنیم
            with st.spinner("🗣️ Transcribing voice..."):
                # Whisper برای تشخیص فرمت به پسوند فایل نیاز دارد
                recorded_audio.name = "audio.wav" 
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=recorded_audio
                )
                final_prompt = transcript.text

        # ادامه منطق در صورتی که ورودی متنی یا صوتی داشته باشیم
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
                    
                    source_docs = retriever.invoke(final_prompt)
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
                    
                    result = llm.invoke(chat_history + [human_msg])
                    answer = result.content
                    
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
            
            # تغییر کلید آپلودرها برای خالی شدن فرم‌ها در پیام‌های بعدی
            st.session_state.file_uploader_key += 1
            st.rerun()
