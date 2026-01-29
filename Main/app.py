import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import sounddevice as sd
import soundfile as sf
import wave
import tempfile
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import noisereduce as nr
from scipy.signal import butter, lfilter
import io
import time

# Set page config
st.set_page_config(
    page_title="🎭 Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .emotion-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.2);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===== AUDIO PROCESSING FUNCTIONS =====

def butter_bandpass(lowcut, highcut, sr, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(audio, lowcut=300, highcut=3400, sr=22050, order=5):
    b, a = butter_bandpass(lowcut, highcut, sr, order=order)
    return lfilter(b, a, audio)

def preprocess_audio(audio, sr, apply_noise_red=True, apply_bandpass=True):
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    if apply_noise_red:
        audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
    
    if apply_bandpass:
        audio = bandpass_filter(audio, sr=sr)
    
    return audio

def extract_features(audio, sr):
    result = np.array([])
    
    # MFCC
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))
    
    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    result = np.hstack((result, chroma))
    
    # Mel-spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    result = np.hstack((result, mel))
    
    # Spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
    result = np.hstack((result, contrast))
    
    # Tonnetz
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)
    result = np.hstack((result, tonnetz))
    
    # Zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    result = np.hstack((result, zcr))
    
    return result

# ===== LOAD MODEL =====

@st.cache_resource
def load_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_labels(labels_path):
    try:
        labels = np.load(labels_path, allow_pickle=True)
        return labels
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return None

# ===== PREDICTION FUNCTION =====

def predict_emotion(audio_data, sr, model, labels):
    try:
        # Preprocess
        audio = preprocess_audio(audio_data, sr)
        
        # Extract features
        features = extract_features(audio, sr)
        features = np.expand_dims(features, axis=0)
        
        # Predict
        prediction = model.predict(features, verbose=0)
        emotion_idx = np.argmax(prediction)
        confidence = prediction[0][emotion_idx]
        emotion = labels[emotion_idx]
        
        # Get all predictions
        all_predictions = [(labels[i], prediction[0][i]) for i in range(len(labels))]
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        
        return emotion, confidence, all_predictions
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

# ===== VISUALIZATION FUNCTIONS =====

def create_emotion_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 75], 'color': '#ffffcc'},
                {'range': [75, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'size': 16}
    )
    
    return fig

def create_prediction_bars(predictions):
    emotions = [p[0].capitalize() for p in predictions]
    confidences = [p[1] * 100 for p in predictions]
    
    colors = px.colors.sequential.Viridis
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidences,
            y=emotions,
            orientation='h',
            marker=dict(
                color=confidences,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f'{c:.1f}%' for c in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="All Emotion Predictions",
        xaxis_title="Confidence (%)",
        yaxis_title="Emotion",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'size': 14}
    )
    
    fig.update_xaxes(range=[0, 100], gridcolor='lightgray')
    
    return fig

def create_waveform(audio_data, sr):
    time_axis = np.linspace(0, len(audio_data) / sr, len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio_data,
        mode='lines',
        line=dict(color='#667eea', width=1),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'size': 12}
    )
    
    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')
    
    return fig

def create_spectrogram(audio_data, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    
    fig = go.Figure(data=go.Heatmap(
        z=D,
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title="Spectrogram",
        xaxis_title="Time",
        yaxis_title="Frequency",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'size': 12}
    )
    
    return fig

# ===== EMOTION EMOJIS =====

EMOTION_EMOJIS = {
    'angry': '😠',
    'disgust': '🤢',
    'fearful': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprised': '😲',
    'calm': '😌'
}

# ===== MAIN APP =====

def main():
    st.markdown('<h1 class="main-header">🎭 Speech Emotion Recognition</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_path = st.text_input("Model Path", value="emotion_model.h5")
        labels_path = st.text_input("Labels Path", value="label_encoder_classes.npy")
        
        st.markdown("---")
        
        st.header("🎚️ Audio Settings")
        duration = st.slider("Recording Duration (s)", 1, 10, 3)
        sample_rate = st.selectbox("Sample Rate", [16000, 22050, 44100], index=1)
        
        apply_noise_reduction = st.checkbox("Apply Noise Reduction", value=True)
        apply_bandpass = st.checkbox("Apply Bandpass Filter", value=True)
        
        st.markdown("---")
        
        st.header("📊 Display Options")
        show_waveform = st.checkbox("Show Waveform", value=True)
        show_spectrogram = st.checkbox("Show Spectrogram", value=True)
        show_all_predictions = st.checkbox("Show All Predictions", value=True)
        
        st.markdown("---")
        st.info("💡 Tip: Speak clearly and expressively for best results!")
    
    # Load model
    model = load_model(model_path)
    labels = load_labels(labels_path)
    
    if model is None or labels is None:
        st.error("❌ Could not load model or labels. Please check the paths.")
        st.stop()
    
    st.success(f"✅ Model loaded successfully! Supports {len(labels)} emotions: {', '.join(labels)}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎤 Live Recording", "📁 Upload File", "📜 History"])
    
    # Tab 1: Live Recording
    with tab1:
        st.header("Record Your Voice")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🎙️ Start Recording", type="primary", use_container_width=True):
                with st.spinner(f"Recording for {duration} seconds..."):
                    # Record audio
                    audio_data = sd.rec(int(duration * sample_rate), 
                                       samplerate=sample_rate, 
                                       channels=1, 
                                       dtype='float32')
                    sd.wait()
                    audio_data = audio_data.flatten()
                    
                    st.success("✅ Recording complete!")
                    
                    # Save to session state
                    st.session_state['audio_data'] = audio_data
                    st.session_state['sample_rate'] = sample_rate
                    st.session_state['timestamp'] = datetime.now()
        
        if 'audio_data' in st.session_state:
            audio_data = st.session_state['audio_data']
            sr = st.session_state['sample_rate']
            
            # Play audio
            st.audio(audio_data, sample_rate=sr)
            
            # Predict emotion
            with st.spinner("Analyzing emotion..."):
                emotion, confidence, all_predictions = predict_emotion(
                    audio_data, sr, model, labels
                )
            
            if emotion:
                # Display result
                emoji = EMOTION_EMOJIS.get(emotion.lower(), '🎭')
                
                st.markdown(f"""
                <div class="emotion-card">
                    <div style="font-size: 4rem;">{emoji}</div>
                    <div>Detected Emotion: {emotion.upper()}</div>
                    <div class="confidence-badge">Confidence: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_emotion_gauge(confidence), use_container_width=True)
                
                with col2:
                    if show_all_predictions:
                        st.plotly_chart(create_prediction_bars(all_predictions), use_container_width=True)
                
                # Audio visualizations
                if show_waveform:
                    st.plotly_chart(create_waveform(audio_data, sr), use_container_width=True)
                
                if show_spectrogram:
                    st.plotly_chart(create_spectrogram(audio_data, sr), use_container_width=True)
                
                # Save to history
                if 'history' not in st.session_state:
                    st.session_state['history'] = []
                
                st.session_state['history'].append({
                    'timestamp': st.session_state['timestamp'],
                    'emotion': emotion,
                    'confidence': confidence,
                    'duration': duration,
                    'source': 'Recording'
                })
    
    # Tab 2: Upload File
    with tab2:
        st.header("Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV, MP3, etc.)",
            type=['wav', 'mp3', 'ogg', 'flac', 'm4a']
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Load audio
            try:
                audio_data, sr = librosa.load(tmp_path, duration=duration, sr=sample_rate)
                
                st.success("✅ File loaded successfully!")
                st.audio(uploaded_file)
                
                # Predict
                with st.spinner("Analyzing emotion..."):
                    emotion, confidence, all_predictions = predict_emotion(
                        audio_data, sr, model, labels
                    )
                
                if emotion:
                    # Display result
                    emoji = EMOTION_EMOJIS.get(emotion.lower(), '🎭')
                    
                    st.markdown(f"""
                    <div class="emotion-card">
                        <div style="font-size: 4rem;">{emoji}</div>
                        <div>Detected Emotion: {emotion.upper()}</div>
                        <div class="confidence-badge">Confidence: {confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(create_emotion_gauge(confidence), use_container_width=True)
                    
                    with col2:
                        if show_all_predictions:
                            st.plotly_chart(create_prediction_bars(all_predictions), use_container_width=True)
                    
                    # Audio visualizations
                    if show_waveform:
                        st.plotly_chart(create_waveform(audio_data, sr), use_container_width=True)
                    
                    if show_spectrogram:
                        st.plotly_chart(create_spectrogram(audio_data, sr), use_container_width=True)
                    
                    # Save to history
                    if 'history' not in st.session_state:
                        st.session_state['history'] = []
                    
                    st.session_state['history'].append({
                        'timestamp': datetime.now(),
                        'emotion': emotion,
                        'confidence': confidence,
                        'duration': duration,
                        'source': f'Upload: {uploaded_file.name}'
                    })
                
                # Cleanup
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Tab 3: History
    with tab3:
        st.header("Prediction History")
        
        if 'history' in st.session_state and len(st.session_state['history']) > 0:
            df = pd.DataFrame(st.session_state['history'])
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", len(df))
            
            with col2:
                avg_confidence = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with col3:
                most_common = df['emotion'].mode()[0]
                st.metric("Most Common", most_common.capitalize())
            
            with col4:
                if st.button("🗑️ Clear History"):
                    st.session_state['history'] = []
                    st.rerun()
            
            st.markdown("---")
            
            # Emotion distribution
            emotion_counts = df['emotion'].value_counts()
            
            fig = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="Emotion Distribution",
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # History table
            st.subheader("📋 Detailed History")
            
            display_df = df.copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
            display_df['emotion'] = display_df['emotion'].apply(lambda x: x.capitalize())
            
            st.dataframe(
                display_df[['timestamp', 'emotion', 'confidence', 'source']],
                use_container_width=True,
                hide_index=True
            )
            
            # Download history
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name=f"emotion_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        else:
            st.info("No predictions yet. Start recording or upload a file!")

if __name__ == "__main__":
    main()
