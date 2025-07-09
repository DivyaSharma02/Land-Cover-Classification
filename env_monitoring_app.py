import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="Environmental Monitoring & Land Cover Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.main-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
}

.main-header {
    font-size: 3rem;
    font-weight: 700;
    color: #000000;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.sub-header {
    font-size: 1.8rem;
    font-weight: 600;
    color: #2c3e50;
    margin-top: 2rem;
    margin-bottom: 1rem;
    position: relative;
    padding-bottom: 0.5rem;
}

.sub-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 60px;
    height: 3px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 2px;
}

.prediction-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 2rem;
    border-radius: 20px;
    margin: 1rem 0;
    color: white;
    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
}

.analysis-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    color: white;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

.analysis-card:hover {
    transform: translateY(-5px);
}

.environmental-insight {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    color: #2c3e50;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    color: white;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
}

.image-analysis-container {
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    padding: 2rem;
    border-radius: 20px;
    margin: 1rem 0;
    color: #8b4513;
    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
}

.sidebar .stButton > button {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    width: 100%;
    margin-bottom: 0.5rem;
}

.upload-area {
    border: 3px dashed #667eea;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%);
    transition: all 0.3s ease;
}

.upload-area:hover {
    border-color: #764ba2;
    background: linear-gradient(135deg, #e6f3ff 0%, #f8f9ff 100%);
}

.stSelectbox > div > div {
    background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%);
    border-radius: 10px;
}

.stTextInput > div > div {
    background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%);
    border-radius: 10px;
}

.sidebar .stMarkdown {
    background: rgba(255, 255, 255, 0.1);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'current_image_info' not in st.session_state:
    st.session_state.current_image_info = None

# Define class names and environmental information
CLASS_NAMES = ['Cloudy', 'Desert', 'Green_Area', 'Water']
CLASS_COLORS = {
    'Cloudy': '#87CEEB',
    'Desert': '#F4A460',
    'Green_Area': '#32CD32',
    'Water': '#1E90FF'
}

ENVIRONMENTAL_INFO = {
    'Cloudy': {
        'description': 'Cloud-covered areas indicating weather patterns and atmospheric conditions',
        'environmental_impact': 'Clouds play a crucial role in Earth\'s climate system by reflecting sunlight and regulating temperature',
        'indicators': ['Weather patterns', 'Precipitation potential', 'Temperature regulation'],
        'color': '#87CEEB',
        'icon': '☁️'
    },
    'Desert': {
        'description': 'Arid or semi-arid regions with limited vegetation and water resources',
        'environmental_impact': 'Deserts are important ecosystems that support unique biodiversity and influence global climate patterns',
        'indicators': ['Low precipitation', 'High temperature variation', 'Limited vegetation'],
        'color': '#F4A460',
        'icon': '🏜️'
    },
    'Green_Area': {
        'description': 'Vegetated areas including forests, grasslands, and agricultural lands',
        'environmental_impact': 'Green areas are vital for carbon sequestration, oxygen production, and biodiversity conservation',
        'indicators': ['Carbon absorption', 'Biodiversity hotspots', 'Air purification'],
        'color': '#32CD32',
        'icon': '🌳'
    },
    'Water': {
        'description': 'Water bodies including rivers, lakes, oceans, and wetlands',
        'environmental_impact': 'Water bodies are essential for life support, climate regulation, and ecosystem services',
        'indicators': ['Water quality', 'Aquatic ecosystems', 'Climate regulation'],
        'color': '#1E90FF',
        'icon': '💧'
    }
}

def load_trained_model():
    """Load the trained model"""
    try:
        model = load_model('Modelenv.v1.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_model_with_compatibility(model_path):
    """Load model with compatibility handling"""
    try:
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        return model
    except Exception as e1:
        try:
            model = load_model(model_path, compile=False, custom_objects={})
            model.compile(optimizer='adam', 
                         loss='categorical_crossentropy', 
                         metrics=['accuracy'])
            return model
        except Exception as e2:
            try:
                model = create_model_architecture()
                model.load_weights(model_path)
                return model
            except Exception as e3:
                st.error(f"All loading methods failed:")
                st.error(f"Method 1: {e1}")
                st.error(f"Method 2: {e2}")
                st.error(f"Method 3: {e3}")
                return None

def create_model_architecture():
    """Create the model architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=(255, 255, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def analyze_image_properties(uploaded_image):
    """Analyze image properties and characteristics"""
    try:
        img = Image.open(uploaded_image)
        
        # Basic image info
        width, height = img.size
        format_type = img.format
        mode = img.mode
        
        # Convert to numpy array for analysis
        img_array = np.array(img)
        
        # Color analysis
        if len(img_array.shape) == 3:
            avg_brightness = np.mean(img_array)
            color_variance = np.var(img_array)
            
            # Calculate dominant colors
            red_avg = np.mean(img_array[:,:,0])
            green_avg = np.mean(img_array[:,:,1])
            blue_avg = np.mean(img_array[:,:,2])
            
            # Texture analysis (simplified)
            gray = np.mean(img_array, axis=2)
            texture_complexity = np.std(gray)
            
            return {
                'dimensions': f"{width} x {height}",
                'format': format_type,
                'mode': mode,
                'brightness': avg_brightness,
                'color_variance': color_variance,
                'dominant_colors': {
                    'red': red_avg,
                    'green': green_avg,
                    'blue': blue_avg
                },
                'texture_complexity': texture_complexity,
                'file_size': len(uploaded_image.getvalue()) / 1024  # KB
            }
        else:
            return None
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return None

def preprocess_image(uploaded_image):
    """Preprocess image for prediction"""
    try:
        if isinstance(uploaded_image, np.ndarray):
            img = Image.fromarray(uploaded_image)
        else:
            img = Image.open(uploaded_image)
        
        img = img.resize((255, 255))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array, img
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

def predict_image(model, img_array):
    """Make prediction on preprocessed image"""
    try:
        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]
        
        return predicted_class, confidence, prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

def create_prediction_chart(probabilities):
    """Create a beautiful prediction chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=CLASS_NAMES,
            y=probabilities,
            marker_color=[CLASS_COLORS[class_name] for class_name in CLASS_NAMES],
            text=[f'{prob:.1%}' for prob in probabilities],
            textposition='auto',
            textfont=dict(color='white', size=14, family='Inter'),
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "🎯 Land Cover Classification Results",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        xaxis_title="Land Cover Types",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        template="plotly_white",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#2c3e50')
    )
    
    return fig

def create_environmental_analysis(predicted_class, confidence, image_info):
    """Create environmental analysis based on prediction"""
    env_info = ENVIRONMENTAL_INFO[predicted_class]
    
    # Environmental impact assessment
    impact_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
    
    # Recommendations based on land cover type
    recommendations = {
        'Cloudy': [
            "Monitor weather patterns for precipitation forecasts",
            "Consider cloud seeding potential for drought-prone areas",
            "Assess impact on solar energy generation"
        ],
        'Desert': [
            "Implement water conservation strategies",
            "Monitor desertification trends in surrounding areas",
            "Consider renewable energy potential (solar/wind)"
        ],
        'Green_Area': [
            "Maintain biodiversity through conservation efforts",
            "Monitor deforestation and land use changes",
            "Implement sustainable forestry practices"
        ],
        'Water': [
            "Monitor water quality and pollution levels",
            "Assess aquatic ecosystem health",
            "Implement water resource management strategies"
        ]
    }
    
    return {
        'type': predicted_class,
        'confidence': confidence,
        'impact_level': impact_level,
        'description': env_info['description'],
        'environmental_impact': env_info['environmental_impact'],
        'indicators': env_info['indicators'],
        'recommendations': recommendations[predicted_class],
        'color': env_info['color'],
        'icon': env_info['icon']
    }

def main():
    # Main container
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🛰️ Environmental Monitoring & Land Cover Classification</h1>', unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("🔧 Configuration")
        st.sidebar.info(f"🤖 TensorFlow Version: {tf.__version__}")
        
        # Model loading section
        with st.sidebar.expander("🚀 Model Settings", expanded=True):
            model_file = st.file_uploader(
                "Upload Model File (.h5)",
                type=['h5'],
                help="Upload your trained Keras model file"
            )
            
            if model_file is not None:
                if st.button("Load Uploaded Model"):
                    with st.spinner("Loading model..."):
                        with open("temp_model.h5", "wb") as f:
                            f.write(model_file.getbuffer())
                        
                        try:
                            st.session_state.model = load_model_with_compatibility("temp_model.h5")
                            if st.session_state.model:
                                st.success("✅ Model loaded successfully!")
                            if os.path.exists("temp_model.h5"):
                                os.remove("temp_model.h5")
                        except Exception as e:
                            st.error(f"❌ Error loading model: {e}")
                            if os.path.exists("temp_model.h5"):
                                os.remove("temp_model.h5")
            
            st.markdown("---")
            
            weights_file = st.file_uploader(
                "Upload Weights File (.h5)",
                type=['h5'],
                help="Upload weights file if full model loading fails",
                key="weights_uploader"
            )
            
            if weights_file is not None:
                if st.button("Load Weights Only"):
                    with st.spinner("Loading weights..."):
                        try:
                            st.session_state.model = create_model_architecture()
                            with open("temp_weights.h5", "wb") as f:
                                f.write(weights_file.getbuffer())
                            st.session_state.model.load_weights("temp_weights.h5")
                            st.success("✅ Weights loaded successfully!")
                            if os.path.exists("temp_weights.h5"):
                                os.remove("temp_weights.h5")
                        except Exception as e:
                            st.error(f"❌ Error loading weights: {e}")
                            if os.path.exists("temp_weights.h5"):
                                os.remove("temp_weights.h5")
            
            st.markdown("---")
            
            model_path = st.text_input("🔗 Model Path", value="Modelenv.v1.h5")
            
            if st.button("Load Local Model"):
                with st.spinner("Loading model..."):
                    try:
                        st.session_state.model = load_model(model_path)
                        st.success("✅ Model loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error loading model: {e}")
            
            st.markdown("---")
            
            if st.button("Create New Model"):
                with st.spinner("Creating model architecture..."):
                    try:
                        st.session_state.model = create_model_architecture()
                        st.success("✅ New model created!")
                        st.warning("⚠️ This model needs training before use.")
                    except Exception as e:
                        st.error(f"❌ Error creating model: {e}")
        
        # Display model status
        if st.session_state.model:
            st.sidebar.success("✅ Model Ready")
        else:
            st.sidebar.warning("⚠️ Model Not Loaded")
        
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.predictions_history = []
            st.session_state.current_image_info = None
            st.success("🧹 History cleared!")
        
        # Main content area
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown('<h2 class="sub-header">📸 Image Upload & Analysis</h2>', unsafe_allow_html=True)
            
            # File upload with custom styling
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "🖼️ Choose a satellite image for analysis",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a satellite image for land cover classification and environmental analysis"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file is not None:
                # Analyze image properties
                image_info = analyze_image_properties(uploaded_file)
                st.session_state.current_image_info = image_info
                
                # Display uploaded image
                st.image(uploaded_file, caption="📷 Uploaded Satellite Image", use_container_width=True)
                
                # Image properties analysis
                if image_info:
                    st.markdown('<div class="image-analysis-container">', unsafe_allow_html=True)
                    st.markdown("### 🔍 Image Properties Analysis")
                    
                    prop_col1, prop_col2, prop_col3 = st.columns(3)
                    
                    with prop_col1:
                        st.markdown(f"""
                        **📐 Dimensions:** {image_info['dimensions']}  
                        **📁 Format:** {image_info['format']}  
                        **💾 Size:** {image_info['file_size']:.1f} KB
                        """)
                    
                    with prop_col2:
                        st.markdown(f"""
                        **🌟 Brightness:** {image_info['brightness']:.1f}  
                        **🎨 Color Variance:** {image_info['color_variance']:.1f}  
                        **🖼️ Texture:** {image_info['texture_complexity']:.1f}
                        """)
                    
                    with prop_col3:
                        dominant_color = max(image_info['dominant_colors'], key=image_info['dominant_colors'].get)
                        st.markdown(f"""
                        **🎯 Dominant Color:** {dominant_color.title()}  
                        **🔴 Red:** {image_info['dominant_colors']['red']:.1f}  
                        **🟢 Green:** {image_info['dominant_colors']['green']:.1f}  
                        **🔵 Blue:** {image_info['dominant_colors']['blue']:.1f}
                        """)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Prediction button
                if st.button("🔍 Analyze Environmental Features", type="primary"):
                    if st.session_state.model is None:
                        st.error("❌ Please load the model first!")
                    else:
                        with st.spinner("🔄 Analyzing environmental features..."):
                            # Preprocess image
                            img_array, processed_img = preprocess_image(uploaded_file)
                            
                            if img_array is not None:
                                # Make prediction
                                predicted_class, confidence, probabilities = predict_image(
                                    st.session_state.model, img_array
                                )
                                
                                if predicted_class is not None:
                                    # Create environmental analysis
                                    env_analysis = create_environmental_analysis(predicted_class, confidence, image_info)
                                    
                                    # Display prediction results
                                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                                    st.markdown(f"### {env_analysis['icon']} Environmental Classification Results")
                                    
                                    result_col1, result_col2 = st.columns(2)
                                    
                                    with result_col1:
                                        st.markdown(f"""
                                        **🎯 Detected Land Cover:** {predicted_class}  
                                        **📊 Confidence Level:** {confidence:.1%}  
                                        **🌍 Impact Assessment:** {env_analysis['impact_level']}
                                        """)
                                    
                                    with result_col2:
                                        st.markdown(f"""
                                        **📝 Description:** {env_analysis['description'][:100]}...  
                                        **🌱 Environmental Role:** Critical ecosystem component
                                        """)
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Create and display prediction chart
                                    chart = create_prediction_chart(probabilities)
                                    st.plotly_chart(chart, use_container_width=True)
                                    
                                    # Add to history
                                    st.session_state.predictions_history.append({
                                        'predicted_class': predicted_class,
                                        'confidence': confidence,
                                        'timestamp': datetime.now(),
                                        'image_info': image_info
                                    })
        
        with col2:
            st.markdown('<h2 class="sub-header">🌍 Environmental Insights</h2>', unsafe_allow_html=True)
            
            # Display environmental analysis if available
            if st.session_state.predictions_history:
                latest_prediction = st.session_state.predictions_history[-1]
                predicted_class = latest_prediction['predicted_class']
                confidence = latest_prediction['confidence']
                
                env_analysis = create_environmental_analysis(predicted_class, confidence, 
                                                           latest_prediction.get('image_info'))
                
                # Environmental impact card
                st.markdown(f"""
                <div class="analysis-card">
                    <h3>{env_analysis['icon']} {env_analysis['type']} Ecosystem</h3>
                    <p style="font-size: 1.1rem; margin-bottom: 1rem;">{env_analysis['description']}</p>
                    <p><strong>Environmental Impact:</strong> {env_analysis['environmental_impact']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Key environmental indicators
                st.markdown('<div class="environmental-insight">', unsafe_allow_html=True)
                st.markdown("### 🔬 Key Environmental Indicators")
                for indicator in env_analysis['indicators']:
                    st.markdown(f"• {indicator}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Recommendations
                st.markdown(f"""
                <div class="analysis-card">
                    <h3>💡 Environmental Recommendations</h3>
                    <ul style="margin: 0; padding-left: 1.5rem;">
                        {''.join([f'<li style="margin-bottom: 0.5rem;">{rec}</li>' for rec in env_analysis['recommendations']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Session statistics
                st.markdown("### 📊 Session Analytics")
                
                total_predictions = len(st.session_state.predictions_history)
                class_distribution = pd.Series([p['predicted_class'] for p in st.session_state.predictions_history]).value_counts()
                
                stat_col1, stat_col2 = st.columns(2)
                
                with stat_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin: 0; font-size: 2rem;">{total_predictions}</h3>
                        <p style="margin: 0; opacity: 0.9;">Images Analyzed</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stat_col2:
                    most_common = class_distribution.index[0] if len(class_distribution) > 0 else "N/A"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin: 0; font-size: 1.5rem;">{most_common}</h3>
                        <p style="margin: 0; opacity: 0.9;">Most Detected</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Distribution chart
                if len(class_distribution) > 0:
                    fig_dist = px.pie(
                        values=class_distribution.values,
                        names=class_distribution.index,
                        title="🌍 Land Cover Distribution",
                        color=class_distribution.index,
                        color_discrete_map=CLASS_COLORS
                    )
                    fig_dist.update_layout(
                        template="plotly_white",
                        height=300,
                        font=dict(family='Inter', color='#2c3e50'),
                        title_font_size=16
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            else:
                st.markdown(f"""
                <div class="environmental-insight">
                    <h3 style="color: #2c3e50; text-align: center;">🌱 Ready for Environmental Analysis</h3>
                    <p style="text-align: center; font-size: 1.1rem; margin-bottom: 1rem;">
                        Upload a satellite image to discover environmental insights and land cover patterns
                    </p>
                    <div style="text-align: center;">
                        <p><strong>🔍 Analysis Features:</strong></p>
                        <p>• Land cover classification</p>
                        <p>• Environmental impact assessment</p>
                        <p>• Conservation recommendations</p>
                        <p>• Ecosystem health indicators</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; margin-top: 2rem; color: white;">
        <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">
            🌍 Environmental Monitoring System
        </h4>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Powered by Deep Learning • Protecting Our Planet Through Technology
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    