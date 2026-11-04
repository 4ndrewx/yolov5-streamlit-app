import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Configuración de la página
st.set_page_config(
    page_title="YOLOv8 Streamlit App",
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


# Cargar modelo de forma limpia con Ultralytics
@st.cache_resource
def load_yolo_model():
  # Utiliza yolov8n (nano) que es ligero, rápido y descarga sin problemas en Streamlit
  model = YOLO("yolov8n.pt")
  return model


with st.spinner("Cargando modelo de detección... por favor espera."):
  model = load_yolo_model()

# --- CONTENIDO PRINCIPAL ---
st.write("Esta aplicación utiliza YOLO para detectar objetos en imágenes capturadas con tu cámara.")
st.text("Capturar imagen")

img_file_buffer = st.camera_input("Take Photo")

if img_file_buffer is not None:
  image = Image.open(img_file_buffer)

  with st.spinner("Procesando detección..."):
    # Realizar inferencia aplicando los parámetros
    results = model(
        image, conf=confidence_threshold, iou=iou_threshold, max_det=max_detections
    )
    result = results[0]

    # Generar la imagen anotada
    annotated_array = result.plot()
    annotated_image = Image.fromarray(
        annotated_array[..., ::-1]
    )  # Convertir BGR a RGB

  col1, col2 = st.columns([2, 1])

  with col1:
    st.image(
        annotated_image,
        caption="Imagen Procesada con Detecciones",
        use_container_width=True,
    )

  with col2:
    st.subheader("Resultados de Detección")
    boxes = result.boxes

    if len(boxes) > 0:
      st.success(f"¡Se encontraron {len(boxes)} objetos!")

      # Extraer nombres y confianzas para mostrar en tabla
      names_list = []
      conf_list = []
      for box in boxes:
        cls_id = int(box.cls[0])
        names_list.append(result.names[cls_id])
        conf_list.append(float(box.conf[0]))

      import pandas as pd

      df_results = pd.DataFrame({"name": names_list, "confidence": conf_list})
      st.dataframe(df_results, use_container_width=True)
    else:
      st.warning(
          "No se detectaron objetos con los umbrales de confianza actuales."
      )
