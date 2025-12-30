"""
Streamlit Demo App - No PyTorch Required
"""
import streamlit as st

st.set_page_config(page_title="Tweet Sentiment Extraction - DEMO", page_icon="🐦")

st.warning("⚠️ **DEMO MODE**: Mock predictions only (no AI model)")

st.title("🐦 Tweet Sentiment Extraction - DEMO")

st.markdown("""
This is a **demo version** for Windows users with PyTorch installation issues.
""")

# Main interface
st.header("📝 Extract Sentiment Phrases")

text_input = st.text_area("Tweet Text", value="I really love this product!", height=150)
sentiment = st.selectbox("Sentiment", ["positive", "negative", "neutral"])

if st.button("🎯 Extract Sentiment Phrase (DEMO)", type="primary"):
    # Simple mock prediction
    if sentiment == "positive":
        prediction = "really love"
    elif sentiment == "negative":
        prediction = "terrible" if "terrible" in text_input.lower() else text_input.split()[0]
    else:
        prediction = " ".join(text_input.split()[:3])
    
    st.success("✅ Extraction Complete! (DEMO)")
    st.subheader("📊 Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Tweet:**")
        st.info(text_input)
    with col2:
        st.markdown("**Extracted Phrase (MOCK):**")
        st.success(prediction)
    
    st.info("💡 **Note**: This is a mock prediction for demonstration purposes only.")

st.markdown("---")
st.markdown("Built for AIE417 - Selected Topics in AI | **DEMO MODE**")
