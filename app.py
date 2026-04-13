import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import os
import glob

# Ensure project root is in path or assumed if run from root
from craft_df.models.craft_df_model import CRAFTDFModel
from craft_df.data.face_detection import FaceDetector
from craft_df.data.dwt_processing import DWTProcessor
from craft_df.utils.config import load_config
from craft_df.data.transforms import get_transforms

st.set_page_config(
    page_title="CRAFT-DF Inference Engine",
    page_icon="🔍",
    layout="wide"
)

# --- 1. Load Resource Caches ---
@st.cache_resource
def init_model(checkpoint_path=None):
    """Load PyTorch Lightning Checkpoint"""
    if checkpoint_path is None:
        checkpoint_path = "checkpoints/remote_ckpt/last.ckpt"
        if not os.path.exists(checkpoint_path):
            return None
        
    config = load_config("configs/default.yaml")
    
    st.sidebar.success(f"Loaded checkpoint: {Path(checkpoint_path).name}")
    
    # Initialize un-compiled model and load weights
    model = CRAFTDFModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Figure out the device
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    model.to(device)
    
    return model, device

@st.cache_resource
def init_processors():
    """Initialize Heavy Dataprep Extractors"""
    face_detector = FaceDetector(min_detection_confidence=0.7)
    dwt_processor = DWTProcessor(wavelet='db4', levels=3)
    _, val_transform = get_transforms(augmentation=False) # Get CombinedTransform
    return face_detector, dwt_processor, val_transform

# --- 2. Title & Descriptions ---
st.title("🔍 CRAFT-DF Live Inference")
st.markdown("""
Upload a facial image and run the Cross-Attentive Frequency-Temporal Disentanglement engine. 
The system will automatically isolate spatial faces and compute Daubechies DWT frequencies before predicting generative artifacts.
""")

# Setup sidebar
st.sidebar.header("Configuration")

with st.spinner("Initializing AI Components..."):
    face_det, dwt_proc, val_transform = init_processors()
    model_data = init_model()

if model_data is None:
    st.error("No tracked checkpoints found! Please run `train.py` to compile model weights natively into `checkpoints/` before running inference.")
    st.stop()
    
model, device = model_data

# --- 3. Interaction Loop ---

uploaded_file = st.file_uploader("Upload target image", type=['png', 'jpg', 'jpeg', 'webp'])

if uploaded_file is not None:
    
    # Load into memory mapping
    pil_image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Source Image")
        st.image(image_np, use_container_width=True)
        
    # Start Prediction Routine
    if st.button("Run Deepfake Detection", type="primary", use_container_width=True):
        
        with st.status("Analyzing Geometry...", expanded=True) as status:
            st.write("Extracting face geometries...")
            
            # Step A. Detect Face
            faces = face_det.extract_faces(image_bgr, max_faces=1)
            
            if not faces:
                status.update(label="Analysis failed", state="error")
                st.error("No faces detected in the image boundary!")
            else:
                face_crop, box_confidence = faces[0]
                
                # Render Crop
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                with col2:
                    st.subheader("2. Extracted Topology")
                    st.image(face_rgb, caption=f"Face Confidence: {box_confidence:.2%}", use_container_width=True)

                st.write("Normalizing tensors...")
                # Step B. Tensor Prep
                try:
                    # 1. Spatial Tensor
                    spatial = torch.from_numpy(face_crop).float() / 255.0
                    if spatial.dim() == 3 and spatial.shape[-1] in (1, 3):
                        spatial = spatial.permute(2, 0, 1) # CHW
                        
                    # 2. DWT Tensor (Model internally computes DWT)
                    freq = spatial.clone()
                    
                    # 3. Apply standard validation transforms
                    spatial, freq, _ = val_transform((spatial, freq, 0)) # Send 0 label dummy
                    
                    # 4. Batch mapping to device
                    spatial_batch = spatial.unsqueeze(0).to(device)
                    freq_batch = freq.unsqueeze(0).to(device)
                    
                    st.write("Streaming through CRAFT-DF Attention networks...")
                    # Step D. Model Assessment
                    with torch.no_grad():
                        outputs = model(spatial_batch, freq_batch)
                        probabilities = outputs['predictions'].cpu().numpy()[0]
                        
                    fake_score = probabilities[1]
                    real_score = probabilities[0]
                    
                    status.update(label="Scan Complete!", state="complete")
                    
                    # Step E. Result visualization
                    with col3:
                        st.subheader("3. Forensic Result")
                        if fake_score > 0.5:
                            st.error(f"⚠️ HIGH RISK OF MANIPULATION\n\nAI Alteration Confidence: {fake_score:.2%}")
                        else:
                            st.success(f"✅ VERIFIED AUTHENTIC\n\nAuthenticity Confidence: {real_score:.2%}")
                            
                        # Build standard progress metrics
                        st.progress(float(fake_score), text="Generative Artifacts Index")

                except Exception as e:
                    status.update(label="Processing fault", state="error")
                    st.error(f"Inference pipeline crashed safely: {str(e)}")
