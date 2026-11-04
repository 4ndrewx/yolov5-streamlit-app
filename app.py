import numpy as np
from PIL import Image
import streamlit as st
import torch

# Configuración de la página en modo oscuro / ancho
st.set_page_config(
    page_title="YOLOv5 Streamlit App",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- BARRA LATERAL: PARÁMETROS ---
st.sidebar.title("Parámetros")
st.sidebar.subheader("Configuración de detección")

confidence_threshold = st.sidebar.slider(
    "Confianza mínima", min_value=0.0, max_value=1.0, value=0.25, step=0.01
)

iou_threshold = st.sidebar.slider(
    "Umbral IoU", min_value=0.0, max_value=1.0, value=0.45, step=0.01
)

max_detections = st.sidebar.slider(
    "Detecciones máximas", min_value=10, max_value=2000, value=1000, step=10
)


# Cargar modelo YOLOv5 desde PyTorch Hub
@st.cache_resource
def load_yolov5_model():
  model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
  return model


with st.spinner(
    "Cargando modelo YOLOv5... por favor espera un momento."
):
  model = load_yolov5_model()

# --- CONTENIDO PRINCIPAL ---
st.write(
    "Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes"
    " capturadas con tu cámara."
)
st.text("Capturar imagen")

# Componente de la cámara
img_file_buffer = st.camera_input("Take Photo")

if img_file_buffer is not None:
  image = Image.open(img_file_buffer)

  # Aplicar parámetros
  model.conf = confidence_threshold
  model.iou = iou_threshold
  model.max_det = max_detections

  with st.spinner("Procesando detección..."):
    results = model(image)
    results.render()
    annotated_image = Image.fromarray(results.ims[0])

  col1, col2 = st.columns([2, 1])

  with col1:
    st.image(
        annotated_image,
        caption="Imagen Procesada con Detecciones",
        use_container_width=True,
    )

  with col2:
    st.subheader("Resultados de Detección")
    df_results = results.pandas().xyxy[0]

    if not df_results.empty:
      st.success(f"¡Se encontraron {len(df_results)} objetos!")
      st.dataframe(df_results[["name", "confidence"]], use_container_width=True)
    else:
      st.warning(
          "No se detectaron objetos con los umbrales de confianza actuales."
      )
