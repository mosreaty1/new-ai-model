"""
Streamlit application for Tweet Sentiment Extraction.
Phase 3: Deployment
"""
import streamlit as st
import torch
from config import ProjectConfig
from model_trainer import SentimentExtractionTrainer
import os


@st.cache_resource
def load_model(model_path: str):
    """Load model with caching."""
    config = ProjectConfig()
    trainer = SentimentExtractionTrainer(config)
    trainer.load_finetuned_model(model_path)
    return trainer, config


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Tweet Sentiment Extraction",
        page_icon="🐦",
        layout="wide"
    )

    # Title and description
    st.title("🐦 Tweet Sentiment Extraction")
    st.markdown("""
    **AIE417 Selected Topics in AI - Fall 2025**

    This application extracts sentiment-bearing phrases from tweets using a fine-tuned language model with PEFT/LoRA.
    """)

    # Sidebar for model configuration
    st.sidebar.header("⚙️ Model Configuration")

    model_path = st.sidebar.text_input(
        "Model Path",
        value="./models/flan-t5-sentiment-extraction",
        help="Path to the fine-tuned model directory"
    )

    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        st.info("Please train the model first using `python train.py`")
        return

    # Load model
    try:
        with st.spinner("Loading model..."):
            trainer, config = load_model(model_path)
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return

    # Inference parameters
    st.sidebar.subheader("Generation Parameters")

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=config.model.temperature,
        step=0.1,
        help="Controls randomness in generation"
    )

    top_p = st.sidebar.slider(
        "Top-p (Nucleus Sampling)",
        min_value=0.1,
        max_value=1.0,
        value=config.model.top_p,
        step=0.05,
        help="Cumulative probability threshold for token selection"
    )

    top_k = st.sidebar.slider(
        "Top-k",
        min_value=1,
        max_value=100,
        value=config.model.top_k,
        step=1,
        help="Number of top tokens to consider"
    )

    num_beams = st.sidebar.slider(
        "Number of Beams",
        min_value=1,
        max_value=10,
        value=config.model.num_beams,
        step=1,
        help="Number of beams for beam search"
    )

    max_new_tokens = st.sidebar.slider(
        "Max New Tokens",
        min_value=16,
        max_value=128,
        value=config.model.max_new_tokens,
        step=8,
        help="Maximum length of generated text"
    )

    # Update config
    config.model.temperature = temperature
    config.model.top_p = top_p
    config.model.top_k = top_k
    config.model.num_beams = num_beams
    config.model.max_new_tokens = max_new_tokens

    # Main interface
    st.header("📝 Extract Sentiment Phrases")

    col1, col2 = st.columns(2)

    with col1:
        text_input = st.text_area(
            "Tweet Text",
            height=150,
            placeholder="Enter a tweet here...",
            help="The tweet from which to extract sentiment"
        )

    with col2:
        sentiment = st.selectbox(
            "Sentiment",
            options=["positive", "negative", "neutral"],
            help="The sentiment to extract"
        )

    # Example tweets
    st.subheader("📋 Example Tweets")
    examples = {
        "Positive": {
            "text": "I really really like the song Love Story by Taylor Swift",
            "sentiment": "positive"
        },
        "Negative": {
            "text": "My boss is bullying me at work and it's making me miserable",
            "sentiment": "negative"
        },
        "Neutral": {
            "text": "I am going to the store to buy some groceries",
            "sentiment": "neutral"
        }
    }

    example_cols = st.columns(3)
    for i, (label, example) in enumerate(examples.items()):
        with example_cols[i]:
            if st.button(f"Use {label} Example"):
                text_input = example["text"]
                sentiment = example["sentiment"]
                st.rerun()

    # Generate prediction
    if st.button("🎯 Extract Sentiment Phrase", type="primary"):
        if not text_input:
            st.warning("Please enter some text!")
        else:
            with st.spinner("Generating prediction..."):
                try:
                    predictions = trainer.generate_predictions(
                        [text_input],
                        [sentiment]
                    )
                    prediction = predictions[0]

                    # Display results
                    st.success("✅ Extraction Complete!")

                    st.subheader("📊 Results")

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        st.markdown("**Original Tweet:**")
                        st.info(text_input)

                        st.markdown("**Sentiment:**")
                        sentiment_color = {
                            "positive": "🟢",
                            "negative": "🔴",
                            "neutral": "⚪"
                        }
                        st.info(f"{sentiment_color[sentiment]} {sentiment.upper()}")

                    with result_col2:
                        st.markdown("**Extracted Phrase:**")
                        st.success(prediction)

                        # Highlight in original text
                        st.markdown("**Highlighted in Text:**")
                        if prediction.lower() in text_input.lower():
                            highlighted = text_input.replace(
                                prediction,
                                f"**:green[{prediction}]**"
                            )
                            st.markdown(highlighted)
                        else:
                            st.markdown(f"{text_input}\n\n*(Extracted: **:green[{prediction}]**)*")

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

    # Batch processing
    st.header("📦 Batch Processing")

    uploaded_file = st.file_uploader(
        "Upload CSV file (must have 'text' and 'sentiment' columns)",
        type=["csv"]
    )

    if uploaded_file is not None:
        import pandas as pd

        df = pd.read_csv(uploaded_file)

        if 'text' not in df.columns or 'sentiment' not in df.columns:
            st.error("CSV must contain 'text' and 'sentiment' columns")
        else:
            st.write(f"Loaded {len(df)} rows")
            st.dataframe(df.head())

            if st.button("🚀 Process Batch"):
                with st.spinner(f"Processing {len(df)} samples..."):
                    predictions = trainer.generate_predictions(
                        df['text'].tolist(),
                        df['sentiment'].tolist()
                    )

                    df['selected_text'] = predictions

                    st.success("✅ Batch processing complete!")
                    st.dataframe(df)

                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )

    # Model information
    with st.expander("ℹ️ Model Information"):
        st.markdown(f"""
        **Base Model:** `{config.model.base_model}`

        **PEFT Configuration:**
        - LoRA r: {config.lora.r}
        - LoRA alpha: {config.lora.lora_alpha}
        - LoRA dropout: {config.lora.lora_dropout}
        - Target modules: {config.lora.target_modules}

        **Training Configuration:**
        - Epochs: {config.training.num_train_epochs}
        - Batch size: {config.training.per_device_train_batch_size}
        - Learning rate: {config.training.learning_rate}
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ for AIE417 - Selected Topics in AI</p>
        <p>Dr. Laila Shoukry | Fall 2025</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
