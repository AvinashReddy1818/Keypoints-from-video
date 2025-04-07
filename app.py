import streamlit as st
import whisper
import google.generativeai as genai
import pandas as pd
import os
import re
import tempfile
import subprocess
import base64
from fuzzywuzzy import fuzz
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import streamlit.components.v1 as components
from io import BytesIO

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Ensure outputs directory exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Convert seconds to hh:mm:ss
def convert_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

# Extract audio
def extract_audio(video_path):
    temp_audio_path = video_path.replace(".mp4", ".wav")
    try:
        command = [
            "ffmpeg", "-i", video_path, "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le", temp_audio_path, "-y"
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return temp_audio_path
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Error extracting audio: {e}")
        return None

# Transcribe using Whisper
def transcribe_audio(video_path, model_size="small"):
    st.info("🔍 Transcribing audio...")
    audio_path = extract_audio(video_path)
    if not audio_path:
        return None
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    transcript = [
        {"start": convert_to_hms(seg["start"]), "end": convert_to_hms(seg["end"]), "text": seg["text"].strip()}
        for seg in result.get("segments", [])]
    os.remove(audio_path)
    st.success("✅ Transcription completed!")
    return pd.DataFrame(transcript)

# Extract key points using Gemini
def extract_key_points(transcript_df):
    st.info("🔍 Extracting key points...")
    text_content = " ".join(transcript_df["text"].tolist())
    max_chunk_size = 3000
    text_chunks = [text_content[i:i + max_chunk_size] for i in range(0, len(text_content), max_chunk_size)]
    key_points = []
    model = genai.GenerativeModel("gemini-1.5-flash")

    for chunk in text_chunks:
        prompt = f"Extract the most important key points from this transcript:\n\n{chunk}"
        try:
            response = model.generate_content(prompt)
            extracted_points = response.text.strip().split("\n")
            key_points.extend([point.strip() for point in extracted_points if point.strip()])
        except Exception as e:
            st.error(f"❌ Error extracting key points: {e}")
    return key_points if key_points else st.warning("⚠️ No key points extracted.")

# Clean text
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text).lower().strip()

# Convert hh:mm:ss to seconds
def time_to_seconds(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s

# Map key points to timestamps
def map_key_points_to_timestamps(transcript_df, key_points):
    st.info("🔍 Mapping key points to timestamps...")
    results = []
    used_texts = set()
    matching_threshold = 75

    for point in key_points:
        clean_point = clean_text(point)
        for _, row in transcript_df.iterrows():
            cleaned_transcript = clean_text(row["text"])
            similarity_score = fuzz.partial_ratio(clean_point, cleaned_transcript)
            if similarity_score >= matching_threshold and row["text"] not in used_texts:
                results.append({
                    "timestamp": f"{row['start']} - {row['end']}",
                    "important_point": point,
                    "start_sec": time_to_seconds(row['start'])
                })
                used_texts.add(row["text"])
                break
    return pd.DataFrame(results) if results else st.warning("⚠️ No key points matched with timestamps.")

# Save to memory (for download button)
def save_results_to_memory(results_df):
    buffer = BytesIO()
    for _, row in results_df.iterrows():
        line = f"{row['timestamp']} - {row['important_point']}\n"
        buffer.write(line.encode('utf-8'))
    buffer.seek(0)
    return buffer

# Auto-save to file in outputs/
def save_results_to_file(results_df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{OUTPUT_DIR}/key_points_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for _, row in results_df.iterrows():
            f.write(f"{row['timestamp']} - {row['important_point']}\n")
    return filename

# Evaluation
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity_to_transcript(key_points, transcript_texts):
    transcript_joined = " ".join(transcript_texts)
    emb_transcript = embed_model.encode(transcript_joined, convert_to_tensor=True)
    similarities = []
    for point in key_points:
        emb_point = embed_model.encode(point, convert_to_tensor=True)
        score = util.cos_sim(emb_point, emb_transcript).item()
        similarities.append(score)
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    return round(avg_similarity, 4)

def diversity_of_keypoints(key_points):
    if len(key_points) < 2:
        return 1.0
    embeddings = embed_model.encode(key_points)
    total_sim = 0
    count = 0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            score = util.cos_sim(embeddings[i], embeddings[j]).item()
            total_sim += score
            count += 1
    avg_sim = total_sim / count if count > 0 else 0
    diversity = 1 - avg_sim
    return round(diversity, 4)

def compression_ratio(key_points, transcript_df):
    return round(len(key_points) / len(transcript_df), 4)

def average_fuzzy_match_score(key_points, transcript_df):
    scores = []
    for point in key_points:
        clean_point = clean_text(point)
        best_score = 0
        for _, row in transcript_df.iterrows():
            cleaned_transcript = clean_text(row["text"])
            score = fuzz.partial_ratio(clean_point, cleaned_transcript)
            best_score = max(best_score, score)
        scores.append(best_score)
    return round(sum(scores) / len(scores), 2) if scores else 0

# Streamlit UI
st.set_page_config(page_title="Video Insights Extractor", layout="wide")
st.title("🎥 Video Insights Extractor")
st.write("Upload a video, extract key points, and click timestamps in the table!")

uploaded_file = st.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file:
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "results_df" not in st.session_state:
        st.session_state.results_df = None
    if "video_path" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            st.session_state.video_path = temp_file.name

    st.info("📌 Video uploaded. Click 'Process Video' to begin analysis.")

    if st.button("🚀 Process Video"):
        st.session_state.processed = False
        transcript_df = transcribe_audio(st.session_state.video_path)
        if transcript_df is not None:
            key_points = extract_key_points(transcript_df)
            results_df = map_key_points_to_timestamps(transcript_df, key_points)

            if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                st.session_state.results_df = results_df
                st.session_state.transcript_df = transcript_df
                st.session_state.key_points = key_points
                st.session_state.processed = True
                saved_path = save_results_to_file(results_df)
                st.success(f"📁 Results saved to: {saved_path}")
            else:
                st.warning("⚠️ Could not extract any results.")
                st.stop()

if st.session_state.get("processed", False) and st.session_state.get("results_df") is not None:
    results_df = st.session_state.results_df
    transcript_df = st.session_state.transcript_df
    key_points = st.session_state.key_points

    with open(st.session_state.video_path, "rb") as f:
        b64_video = base64.b64encode(f.read()).decode()

    video_html = f"""
    <div style="display:flex; justify-content:center; margin-bottom: 1rem;">
        <video id="customVideo" width="800" height="450" controls>
            <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
            Your browser does not support HTML5 video.
        </video>
    </div>
    """

    table_html = f"""
<style>
    table {{
        width: 100%;
        border-collapse: collapse;
        color: white;
        table-layout: fixed;
        word-wrap: break-word;
    }}
    th, td {{
        border: 1px solid #ccc;
        padding: 10px;
        text-align: left;
        vertical-align: top;
        word-wrap: break-word;
        white-space: normal;
    }}
    td a {{
        color: white;
        text-decoration: underline;
    }}
    tr:hover {{
        background-color: #444;
    }}
</style>
<table>
    <thead>
        <tr>
            <th style="width: 20%;">Timestamp</th>
            <th>Key Point</th>
        </tr>
    </thead>
    <tbody>
"""

    for _, row in results_df.iterrows():
        time_sec = row['start_sec']
        timestamp = row['timestamp']
        point = row['important_point']
        table_html += f"""
            <tr>
                <td><a href="javascript:void(0);" onclick="seekTo({time_sec})">{timestamp}</a></td>
                <td>{point}</td>
            </tr>
        """
    table_html += "</tbody></table>"

    js_script = """
    <script>
    function seekTo(seconds) {
        const video = document.getElementById('customVideo');
        if (video) {
            video.currentTime = seconds;
            video.play();
        }
    }
    </script>
    """

    components.html(video_html + table_html + js_script, height=900, scrolling=False)

    # Download
    txt_buffer = save_results_to_memory(results_df)
    st.download_button("📥 Download Results", data=txt_buffer, file_name="key_points.txt", mime="text/plain")

    # Evaluation
    st.subheader("📊 No-Reference Evaluation")
    sim_score = semantic_similarity_to_transcript(key_points, transcript_df["text"].tolist())
    diversity_score = diversity_of_keypoints(key_points)
    compression = compression_ratio(key_points, transcript_df)
    avg_fuzzy = average_fuzzy_match_score(key_points, transcript_df)

    st.metric("Semantic Similarity", sim_score)
    st.metric("Key Point Diversity", diversity_score)
    st.metric("Compression Ratio", compression)
    st.metric("Avg. Fuzzy Match", avg_fuzzy)