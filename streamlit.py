 #!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# === Configuración inicial (primera línea Streamlit) ===
st.set_page_config(
    page_title="🎗️ Diagnóstico Inteligente Cáncer de Mama",
    layout="centered",
    page_icon="🎗️"
)

# === Inyectar FontAwesome y estilos CSS ===
st.markdown("""
<!-- FontAwesome 6 CDN -->
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet"/>

<style>
/* Ajuste para que no haya scroll extra */
html, body, .main {
    height: 100%;
    margin: 0;
    overflow-y: auto !important;
}

/* Botón flotante */
.boton-flotante {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: linear-gradient(135deg, #f8bbd0cc, #f48fb1cc);
    color: #4a148c !important;
    font-weight: 700;
    font-size: 14px;
    padding: 10px 22px;
    border-radius: 25px;
    text-align: center;
    text-decoration: none;
    cursor: pointer;
    user-select: none;
    box-shadow: 0 0 8px #f48fb199;
    transition: box-shadow 0.3s ease, transform 0.2s ease, background-color 0.3s ease;
    z-index: 9999;
}
.boton-flotante:hover {
    text-decoration: none;
    box-shadow: 0 0 18px #f8bbd0ff, 0 0 28px #f48fb1ff;
    transform: scale(1.08);
    background: linear-gradient(135deg, #f8bbd0ff, #f48fb1ff);
}

/* Footer estilizado */
.footer {
    text-align: center;
    padding: 10px;
    color: gray;
    font-size: 0.9em;
    margin-top: 30px;
    border-top: 1px solid #eee;
}

/* Slider label en negrita */
.stSlider > label {
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# === Variables y etiquetas ===
variables_top10 = {
    "area_mean": "Área promedio de la célula",
    "perimeter_mean": "Perímetro promedio del núcleo",
    "radius_mean": "Radio promedio de la célula",
    "concave points_worst": "Puntos cóncavos máximos",
    "concavity_worst": "Concavidad máxima",
    "compactness_mean": "Compacidad promedio",
    "area_worst": "Área máxima de la célula",
    "radius_worst": "Radio máximo de la célula",
    "concave points_mean": "Puntos cóncavos promedio",
    "perimeter_worst": "Perímetro máximo del núcleo"
}

# === Cargar modelo ===
model_path = "model/random_forest_top10.pkl"
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"🚨 Error cargando modelo en '{model_path}': {e}")
    st.stop()

# === Interfaz principal ===
st.markdown(
    '<h1><i class="fa-solid fa-ribbon" style="color:#ba68c8;"></i> Sistema Predictivo Avanzado de Cáncer de Mama</h1>',
    unsafe_allow_html=True
)
st.markdown("""
Este sistema predictivo, basado en <b>Machine Learning</b>, analiza 10 métricas clínicas clave  
para estimar si un tumor es <b>benigno</b> o <b>maligno</b> con alta precisión.<br><br>
<i class="fa-solid fa-brain" style="color:#7e57c2;"></i> Entrenado con datos reales y validado con técnicas modernas.<br>
<i class="fa-solid fa-chart-line" style="color:#7e57c2;"></i> Precisión alcanzada: <b>94.74%</b>
---""", unsafe_allow_html=True)

# Rango real calculado a partir del dataset original
rangos_variables = {
    "area_mean": (143.5, 2501.0),
    "perimeter_mean": (43.79, 188.5),
    "radius_mean": (6.98, 28.11),
    "concave points_worst": (0.0, 0.291),
    "concavity_worst": (0.0, 1.252),
    "compactness_mean": (0.019, 0.345),
    "area_worst": (185.2, 4254.0),
    "radius_worst": (7.93, 36.04),
    "concave points_mean": (0.0, 0.201),
    "perimeter_worst": (50.41, 251.2)
}

# === Entradas numéricas ===
user_input = {}
cols = st.columns(2)  # dos columnas
for i, (var, label) in enumerate(variables_top10.items()):
    min_val, max_val = rangos_variables[var]
    with cols[i % 2]:
        user_input[var] = st.number_input(
            f"{label} ({var})",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float((min_val + max_val) / 2),
            step=0.01,
            format="%.3f"
        )

input_df = pd.DataFrame([user_input])

# === Botón de diagnóstico ===
if st.button("🚀 Realizar Diagnóstico", type="primary"):
    with st.spinner("🔬 Analizando tus datos..."):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

    st.markdown("## 📋 Resultado del Diagnóstico")

    if prediction == 1:
        st.markdown(
            f"""<p style="color:#FFD43B; font-weight:bold; font-size:18px;">
            <i class="fa-solid fa-triangle-exclamation" style="color: #FFD43B;"></i> 
            Alerta: El tumor podría ser <strong>Maligno</strong> con una probabilidad del {proba*100:.1f}%
            </p>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""<p style="color:#63E6BE; font-weight:bold; font-size:18px;">
            <i class="fa-solid fa-check-double" style="color: #63E6BE;"></i> 
            El tumor podría ser <strong>Benigno</strong> con una probabilidad del {(1-proba)*100:.1f}%
            </p>""",
            unsafe_allow_html=True
        )

    # Gráfico de importancia
    if hasattr(model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Variable": list(variables_top10.keys()),
            "Importancia": model.feature_importances_
        }).sort_values(by="Importancia", ascending=True)

        fig = px.bar(
            fi_df,
            x="Importancia",
            y="Variable",
            orientation="h",
            title='<i class="fa-solid fa-square-poll-horizontal" style="color: #FFD43B;"></i> Importancia de las Variables en la Predicción',
            text=fi_df["Importancia"].apply(lambda x: f"{x:.2f}")
        )
        fig.update_layout(yaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.caption("⚠️ Esta predicción es orientativa y no reemplaza una evaluación médica profesional.")

# === Footer personalizado ===
st.markdown(
    '<div class="footer">Desarrollado por <strong>Vera Martín</strong> &nbsp;|&nbsp; © 2025 Todos los derechos reservados</div>',
    unsafe_allow_html=True
)

# === Botón flotante ===
url_web = "https://tu-pagina-web.com"  # Cambia por tu URL real
st.markdown(
    f'<a href="{url_web}" target="_blank" class="boton-flotante"><i class="fa-solid fa-globe"></i> Visita nuestra Web</a>',
    unsafe_allow_html=True
)