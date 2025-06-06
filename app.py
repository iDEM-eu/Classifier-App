import streamlit as st
import pandas as pd
import plotly.express as px
from model import MODEL_NAMES, predict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from xai import XAI
import os
from IPython.display import display, HTML
from collections import defaultdict
st.set_page_config(layout="wide")
# Function to load CSS
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the external CSS file
css_path = "styles.css"
if os.path.exists(css_path):
    load_css(css_path)

# ---- Header ----

st.markdown("<h1 style='text-align: center;'>Sentence Complexity and XAI</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---- Layout: Input on the Left, Results on the Right ----
col1, col2 = st.columns([1, 2])

with col1:
    ### **🔹 Model Selection & User Input Section**
    st.subheader("⚙️ Model Selection & Settings")
    option = st.radio("Choose an option:", ["Select a single model", "Compare all models"])
    if option == "Select a single model":
        selected_model = st.selectbox("Choose a model:", MODEL_NAMES)
    
    input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"])
    
    if input_method == "Paste Text":
        text = st.text_area("✍️ Enter text:", "", height=150)
    else:
        file = st.file_uploader("📂 Upload a text file", type=["txt"])
    
    explain_xai = st.checkbox("🔎 Explain Predictions (Captum XAI)")
    predict_clicked = st.button("🔍 Analyse")

st.markdown("---")  # **Add a separator between input and output sections**

with col2:
    ### **🔹 Results Section**
    ## st.subheader("📊 Prediction Results")

    # ---- Initialize Variables ----
    simple_count, complex_count = 0, 0
    df_results = pd.DataFrame()

    if predict_clicked:
        results = []
        if option == "Compare all models":
            if input_method == "Paste Text" and text.strip():
               ## st.subheader("📊 Model Predictions")
                progress_bar = st.progress(0)
    
                for i, model_name in enumerate(MODEL_NAMES):
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    model.eval()
    
                    inputs = tokenizer(text, return_tensors="pt", truncation=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        predicted_class = outputs.logits.argmax().item()
    
                    label = "Simple" if predicted_class == 0 else "Complex"
                    if label == "Simple":
                        simple_count += 1
                    else:
                        complex_count += 1
    
                    results.append({"Model": model_name, "Predicted Class": label})
                    progress_bar.progress((i + 1) / len(MODEL_NAMES))
    
                df_results = pd.DataFrame(results)
                st.dataframe(df_results)
                
                # **Insights Section**
                ##st.subheader("📊 Insights")
                if simple_count > complex_count:
                    st.success(f"✅ Majority of models classified the text as **Simple** ({simple_count}/{len(MODEL_NAMES)}).")
                elif complex_count > simple_count:
                    st.error(f" Majority of models classified the text as **Complex** ({complex_count}/{len(MODEL_NAMES)}).")
                else:
                    st.info("⚖️ Equal split between 'Simple' and 'Complex' classifications.")
                    # **Charts and Insights**
                if not df_results.empty:
                    st.subheader("📈 Classification Breakdown")
        
                    # **Improved Bar Chart**
                    fig_bar = px.bar(
                        x=["Simple", "Complex"],
                        y=[simple_count, complex_count],
                        text=[simple_count, complex_count],
                        labels={"x": "Prediction", "y": "Number of Models"},
                        title="How Many Models Classified as Simple vs. Complex",
                        color=["Simple", "Complex"],
                        color_discrete_sequence=["#2ECC71", "#E74C3C"],
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        
                    # **Pie Chart for Class Proportions**
                    fig_pie = px.pie(
                        names=["Simple", "Complex"],
                        values=[simple_count, complex_count],
                        title="Proportion of Simple vs. Complex Predictions",
                        hole=0.3,
                        color_discrete_sequence=["#3498DB", "#E67E22"],
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        
                    # **Sunburst Chart for Model Breakdown**
                    fig_sunburst = px.sunburst(
                        df_results, path=["Predicted Class", "Model"],
                        title="Hierarchy of Model Predictions",
                        color="Predicted Class",
                        color_discrete_map={"Simple": "#2ECC71", "Complex": "#E74C3C"},
                    )
                    st.plotly_chart(fig_sunburst, use_container_width=True)

            else:
                st.warning("⚠️ Please paste some text to compare across models.")

        
        elif input_method == "Paste Text" and text.strip():
            sentences = [text]
            if option == "Select a single model":
               # st.subheader("🔹 Model Prediction")
                model_name_short = selected_model.split("/")[-1]  # Extracts the last part of the model path
                #st.info(f"🧠 **Model Used:** `{model_name_short}`")

                tokenizer, model = AutoTokenizer.from_pretrained(selected_model), AutoModelForSequenceClassification.from_pretrained(selected_model)
                model.eval()

                inputs = tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_class = outputs.logits.argmax().item()

                label = "Simple" if predicted_class == 0 else "Complex"
                st.markdown(f'<div class="result-box">**Predicted Class:** {label}</div>', unsafe_allow_html=True)
                if label == "Complex":
                    #st.subheader(" Advanced Complexity Analysis")

                    with st.spinner("🔄 Running secondary classification..."):
                        typology_model_name = "hannah-khallaf/Typlogy-Classifier"
                        typology_tokenizer = AutoTokenizer.from_pretrained(typology_model_name)
                        typology_model = AutoModelForSequenceClassification.from_pretrained(typology_model_name)
                        typology_model.eval()

                        typology_inputs = typology_tokenizer(text, return_tensors="pt")
                        with torch.no_grad():
                            typology_outputs = typology_model(**typology_inputs)
                            complexity_label = typology_outputs.logits.argmax().item()

                        # Mapping complexity types 
                        complexity_types =  {0: 'Compression',  1:'Explanation',  2: 'Modulation',  3: 'Omission', 4: 'Synonymy',  5: 'Syntactic Changes',  6: 'Transcript',  7: 'Transposition'}

                        detailed_label = complexity_types.get(complexity_label, "Unknown Complexity Type")
                        st.warning(f"**Complexity Type:** {detailed_label}")


                if explain_xai:
                    st.subheader("🔍 Integrated Gradients Explainability")
                    with st.spinner("Computing attributions..."):
                        xai = XAI(text, label, tokenizer, model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                        html_output, top_attributions = xai.generate_html()
                    st.write(HTML(html_output), unsafe_allow_html=True)
                    st.write("📋 **Top Attributed Words**")
                    st.dataframe(top_attributions)

                    
        elif input_method == "Upload File" and file is not None:
            with st.spinner("Processing file..."):
                sentences = [s.strip() for s in file.read().decode("utf-8").split("\n") if s.strip()]

                if not sentences:
                    st.warning("⚠️ The file appears to be empty!")
                else:
                    with st.spinner("🔄 Loading model..."):
                        tokenizer, model = AutoTokenizer.from_pretrained(selected_model), AutoModelForSequenceClassification.from_pretrained(selected_model)
                        model.eval()

                        if explain_xai:
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
                    results = []

                    with st.spinner("Analyzing..."):
                        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
                        
                        with torch.no_grad():
                            outputs = model(**inputs)
                            predicted_classes = outputs.logits.argmax(dim=1).tolist()

                        for i, sentence in enumerate(sentences):
                            label = "Simple" if predicted_classes[i] == 0 else "Complex"
                            complexity_type = "N/A"
                            top_attributed_words = "N/A"

                            # If the sentence is complex, run secondary classification
                            if label == "Complex":
                                with st.spinner("🔄 Running secondary classification..."):
                                    typology_model_name = "hannah-khallaf/Typlogy-Classifier"
                                    typology_tokenizer = AutoTokenizer.from_pretrained(typology_model_name)
                                    typology_model = AutoModelForSequenceClassification.from_pretrained(typology_model_name)
                                    typology_model.eval()

                                typology_inputs = typology_tokenizer(sentence, return_tensors="pt")
                                with torch.no_grad():
                                    typology_outputs = typology_model(**typology_inputs)
                                    complexity_label = typology_outputs.logits.argmax().item()

                                # Mapping complexity types
                                complexity_types = {0: 'Compression', 1: 'Explanation', 2: 'Modulation', 3: 'Omission', 
                                                    4: 'Synonymy', 5: 'Syntactic Changes', 6: 'Transcript', 7: 'Transposition'}
                                complexity_type = complexity_types.get(complexity_label, "Unknown Complexity Type")

                            # If XAI is enabled, run Captum analysis
                            if explain_xai and label == "Complex":
                                xai = XAI(sentence, label, tokenizer, model, device)
                                _, top_attributions = xai.generate_html()
                                top_attributed_words = ", ".join(top_attributions.head(5)['Word'].tolist())

                            results.append({
                                "Sentence": sentence,
                                "Predicted Class": label,
                                "Complexity Type": complexity_type,
                                "Top Attributed Words": top_attributed_words
                            })

                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results)

                    # Provide download link
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download Results", data=csv, file_name="classified_sentences.csv", mime="text/csv")


        
    

    else:
        st.warning("⚠️ Please enter some text to classify.")
        sentences = []
# ---- Footer ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center;">
        <p>🔗 </p>
    </div>
""", unsafe_allow_html=True)
