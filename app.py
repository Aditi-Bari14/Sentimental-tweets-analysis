import streamlit as st
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Streamlit Page Config
st.set_page_config(page_title="Tweet Sentiment Analysis", page_icon="💬", layout="wide")

# Custom CSS for a sleek, modern look
st.markdown("""
    <style>
    /* Background & Text */
    .main { background-color: #f4f6f7; }
    h1, h2, h3 { color: #2E4053; text-align: center; }

    /* Buttons */
    .stButton>button { 
        background: linear-gradient(135deg, #007BFF 30%, #6610f2 100%);
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 12px;
        width: 100%;
        border: none;
    }
    .stFileUploader>div>button {
        background: linear-gradient(135deg, #28a745 30%, #218838 100%);
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 12px;
    }

    /* DataFrame Styling */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #2C3E50 30%, #34495E 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Sentiment Analysis Function
def analyze_sentiment(text):
    if pd.isna(text):
        return "Neutral"
    scores = sia.polarity_scores(text)
    return "Positive" if scores['compound'] >= 0.05 else "Negative" if scores['compound'] <= -0.05 else "Neutral"

# Process CSV
def process_csv(file):
    df = pd.read_csv(file)
    if 'tweet' not in df.columns:
        st.error("CSV file must contain a 'tweet' column")
        return None
    df['Sentiment'] = df['tweet'].apply(analyze_sentiment)
    return df

# Sentiment Distribution Charts
def plot_sentiment_distribution(df):
    sentiment_counts = df['Sentiment'].value_counts()

    # 📊 Beautiful Bar Chart
    # Beautiful Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["#28a745", "#dc3545", "#6c757d"], ax=ax)
    ax.set_title("Sentiment Distribution", fontsize=18, fontweight="bold", color="#2E4053")
    ax.set_xlabel("Sentiment", fontsize=14, color="#2E4053")
    ax.set_ylabel("Count", fontsize=14, color="#2E4053")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    for i, v in enumerate(sentiment_counts.values):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=14, fontweight="bold", color="black")
    st.pyplot(fig)

    # Smaller Pie Chart
    fig_pie, ax_pie = plt.subplots(figsize=(9, 5))  # Same as bar chart
    colors = ["#28a745", "#dc3545", "#6c757d"]
    ax_pie.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors, startangle=140, textprops={'fontsize': 10})
    ax_pie.set_title("Sentiment Proportion", fontsize=12, fontweight="bold", color="#2E4053")
    st.pyplot(fig_pie)
# Sidebar
st.sidebar.markdown("""
    <div style="background-color: #2C3E50; padding: 15px; border-radius: 10px;">
        <h4 style="color: #FFFFFF; text-align: center;">⚙️  Analysis Settings</h4>
        <p style="color: #D0D3D4; text-align: center; font-size: 14px;"></p>
    </div>
""", unsafe_allow_html=True)

# Main UI
st.title("💬 Tweet Sentiment Analysis")

uploaded_file = st.file_uploader("📂 Upload a CSV file with tweets", type=["csv"])

if uploaded_file:
    df = process_csv(uploaded_file)
    if df is not None:
        st.write("### 📝 Processed Data:")
        st.dataframe(df, height=300, width=900)

        st.write("### 📊 Sentiment Analysis Results")
        plot_sentiment_distribution(df)

        # 🔍 Sentiment Filtering with Toggle
        st.write("### 🔍 Filter by Sentiment")
        sentiment_filter = st.selectbox("Choose sentiment", ["All", "Positive", "Negative", "Neutral"])
        filtered_df = df if sentiment_filter == "All" else df[df["Sentiment"] == sentiment_filter]
        st.write(filtered_df)

        # ⬇️ Download Button
        st.download_button("⬇️ Download Processed CSV", df.to_csv(index=False), "processed_tweets.csv", "text/csv")

# Sidebar Styling
st.sidebar.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #2C3E50, #34495E);
        color: white;
        padding: 20px;
    }
    
    .sidebar-title {
        font-size: 22px; 
        font-weight: bold; 
        color: #FFFFFF; 
        text-align: center; 
        margin-bottom: 10px;
    }

    .sidebar-section {
        background-color: rgba(255, 255, 255, 0.1); 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 15px;
    }

    .sidebar-text {
        font-size: 16px; 
        color: #D0D3D4; 
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Content
#st.sidebar.markdown('<div class="sidebar-title">⚙️ Analysis Settings</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section"><p class="sidebar-text">Upload a CSV file and customize analysis options.</p></div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section"><p class="sidebar-text">Choose sentiment filter for refined insights.</p></div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section"><p class="sidebar-text"> Adjust options for better insights.</p></div>', unsafe_allow_html=True)
