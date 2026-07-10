import os
import sys
import time
import tempfile
import streamlit as st

# Ensure root directory is in sys.path
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.settings import get_config
from src.orchestration.pipeline import PipelineOrchestrator
from src.providers.factory import ProviderFactory
from src.shared.providers import (
    LegacyVisionProviderAdapter,
    LegacyAudioProviderAdapter,
    LegacyOCRProviderAdapter,
    LegacyLLMProviderAdapter,
)

# Set up page styling and config
st.set_page_config(
    page_title="ChronosCap - Video Captioning Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium glassmorphism dark-mode look
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    .stSidebar {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }
    h1, h2, h3 {
        color: #6366f1 !important;
        font-family: 'Outfit', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        color: white !important;
    }
    .caption-box {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        backdrop-filter: blur(8px);
    }
    .style-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .badge-formal { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-sarcastic { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-tech { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-nontech { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 ChronosCap")
st.caption("Multi-Modal Semantic Video Captioner Agent Dashboard")

# Sidebar Configuration
st.sidebar.image("cover.jpeg", use_column_width=True)
st.sidebar.markdown("### ⚙️ Pipeline Settings")

# Choice of execution mode
execution_mode = st.sidebar.selectbox(
    "Execution Mode",
    options=["BALANCED", "FAST", "DEEP"],
    index=0,
    help="FAST: Skips speech & OCR, reduces frame budget. BALANCED: Full perception with edge filters. DEEP: Strict verification and maximum retries."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 API Keys Override")
st.sidebar.info("By default, the application will use the API keys defined in your local `.env` file securely. Use fields below to override them if needed.")

groq_key_override = st.sidebar.text_input("Groq API Key", type="password", help="Leave blank to use key from .env")

# Apply API key overrides to environment
if groq_key_override:
    os.environ["GROQ_API_KEY"] = groq_key_override

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Pipeline Modalities")
st.sidebar.checkbox("Visual Model (Llama-3.2-Vision)", value=True, disabled=True)
st.sidebar.checkbox("Speech-To-Text (Whisper-v3)", value=True, disabled=True)
st.sidebar.checkbox("OCR Text Extractor (EasyOCR)", value=True, disabled=True)

# Main Workspace Layout
uploaded_file = st.file_uploader("Upload video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # 1. Save uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎥 Source Video")
        st.video(tfile.name)
        st.info(f"Filename: `{uploaded_file.name}`")

    with col2:
        st.subheader("⚙️ Processing Controls")
        
        # Initialize pipeline on button click
        if st.button("🚀 Run Captioning Agent"):
            status_container = st.container()
            progress_bar = st.progress(0)
            
            try:
                # Load configuration and initialize providers
                with status_container:
                    st.write("🔄 Loading configuration and settings...")
                config = get_config()
                progress_bar.progress(10)

                with status_container:
                    st.write("🔄 Initializing Multi-Modal Providers...")
                
                vision_provs = config.models.vision.provider
                speech_provs = config.models.speech.provider
                ocr_provs = config.models.ocr.provider
                llm_provs = config.models.llm.provider

                raw_vision = ProviderFactory.get_vision(vision_provs)
                raw_speech = ProviderFactory.get_speech(speech_provs)
                raw_ocr = ProviderFactory.get_ocr(ocr_provs)
                raw_llm = ProviderFactory.get_llm(llm_provs)

                vision = LegacyVisionProviderAdapter(raw_vision)
                audio = LegacyAudioProviderAdapter(raw_speech)
                ocr = LegacyOCRProviderAdapter(raw_ocr)
                llm = LegacyLLMProviderAdapter(raw_llm)

                orchestrator = PipelineOrchestrator(
                    config=config,
                    llm_provider=llm,
                    vision_provider=vision,
                    audio_provider=audio,
                    ocr_provider=ocr
                )
                progress_bar.progress(25)

                # Process the video file
                with status_container:
                    st.write(f"🏃 Running Multi-Modal Pipeline in **{execution_mode}** mode...")
                
                # We time the execution
                start_time = time.time()
                captions = orchestrator.process_video(tfile.name, mode=execution_mode)
                elapsed = time.time() - start_time
                
                progress_bar.progress(100)
                with status_container:
                    st.success(f"✅ Processing complete in {elapsed:.2f}s!")

                # Display Captions
                st.subheader("📝 Generated Captions")

                style_classes = {
                    "formal": "badge-formal",
                    "sarcastic": "badge-sarcastic",
                    "humorous_tech": "badge-tech",
                    "humorous_non_tech": "badge-nontech"
                }

                # Internal mapping names used inside the codebase
                internal_mapping = {
                    "formal": "formal",
                    "sarcastic": "sarcastic",
                    "humorous_tech": "tech_humor",
                    "humorous_non_tech": "non_tech_humor"
                }

                for req_style, internal_name in internal_mapping.items():
                    cap_obj = captions.get(internal_name)
                    text = cap_obj.text.strip() if cap_obj else "No caption generated."
                    
                    st.markdown(
                        f"""
                        <div class="caption-box">
                            <span class="style-badge {style_classes[req_style]}">{req_style.replace('_', ' ')}</span>
                            <p style="font-size: 1.15em; line-height: 1.6; color: #f3f4f6; margin-top: 5px;">{text}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Show Explainability EJR
                if "formal" in captions and "ejr" in captions["formal"].metadata:
                    with st.expander("🔍 Show Evidence Justification Record (EJR)"):
                        st.markdown(captions["formal"].metadata["ejr"])

            except Exception as e:
                progress_bar.empty()
                with status_container:
                    st.error(f"❌ Pipeline failed: {e}")
                    st.exception(e)
            
    # Clean up temp file
    try:
        os.unlink(tfile.name)
    except Exception:
        pass
else:
    # Decorative landing information when no video uploaded
    st.markdown("---")
    st.subheader("💡 Features included")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            ### 🎥 Sampling & Ingestion
            Uses PyAV keyframe analysis to estimate scene changes and dynamic target frame rates under token limits.
            """
        )
    with col2:
        st.markdown(
            """
            ### 🧠 Multi-Modal Fusion
            Fuses observations from LLM vision description, local EasyOCR text extraction, and audio transcription.
            """
        )
    with col3:
        st.markdown(
            """
            ### 🤖 Style Generation
            Generates 4 distinct styled captions (Formal, Sarcastic, Humorous Programming/Tech, Humorous Everyday Life) with semantic validation.
            """
        )
