import streamlit as st
import pickle
import re

@st.cache_resource
def load_model():
    with open('ddc_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

DDC_CLASSES = {
    0: "000 - Computer Science, Information & General Works",
    100: "100 - Philosophy & Psychology",
    200: "200 - Religion",
    300: "300 - Social Sciences",
    400: "400 - Language",
    500: "500 - Science",
    600: "600 - Technology",
    700: "700 - Arts & Recreation",
    800: "800 - Literature",
    900: "900 - History & Geography"
}

st.set_page_config(page_title="DDC Classifier", layout="wide")

st.title("Dewey Decimal Classification (DDC) Predictor")
st.markdown("Classify books into DDC main classes based on title and description")

model, vectorizer = load_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Book Information")
    title = st.text_input("Book Title", placeholder="Enter book title...")
    description = st.text_area("Book Description", height=200, 
                               placeholder="Enter book description...")
    
    predict_button = st.button("Predict DDC Class", type="primary", use_container_width=True)

with col2:
    st.subheader("Prediction Result")
    
    if predict_button:
        if not title.strip() or not description.strip():
            st.error("Please enter both title and description")
        else:
            combined_text = title + " " + title + " " + description
            processed_text = preprocess(combined_text)
            
            text_vec = vectorizer.transform([processed_text])
            prediction = model.predict(text_vec)[0]
            
            decision_scores = model.decision_function(text_vec)[0]
            
            classes = model.classes_
            scores_dict = dict(zip(classes, decision_scores))
            sorted_classes = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            
            st.success(f"**Predicted Class: {prediction}**")
            st.info(DDC_CLASSES[prediction])
            
            st.subheader("Confidence Scores")
            st.markdown("Top 3 most likely classes:")
            
            for i, (cls, score) in enumerate(sorted_classes[:3], 1):
                normalized_score = (score - min(decision_scores)) / (max(decision_scores) - min(decision_scores) + 1e-10)
                confidence = normalized_score * 100
                
                st.write(f"**{i}. Class {cls}** - {DDC_CLASSES[cls]}")
                st.progress(confidence / 100)
                st.write(f"Confidence: {confidence:.2f}%")
                st.write("")

st.markdown("---")
st.markdown("### About DDC Classes")

col_a, col_b = st.columns(2)
with col_a:
    for cls in [0, 100, 200, 300, 400]:
        st.write(f"**{cls}**: {DDC_CLASSES[cls].split(' - ')[1]}")

with col_b:
    for cls in [500, 600, 700, 800, 900]:
        st.write(f"**{cls}**: {DDC_CLASSES[cls].split(' - ')[1]}")

st.markdown("---")
st.caption("Model: Linear SVC with TF-IDF | Accuracy: ~68.77%")
