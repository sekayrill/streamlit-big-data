import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import re

# Set page config
st.set_page_config(
    page_title="App Store Reviews Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
    st.session_state.data_generated = False

def generate_dataset(n_samples=3000):
    """Generate realistic app store review dataset"""
    random.seed(42)
    np.random.seed(42)
    
    # Master data
    app_categories = ['Games', 'Social', 'Productivity', 'Entertainment', 'Business', 
                     'Education', 'Health & Fitness', 'Finance', 'News', 'Travel']
    
    # Sentiment words
    positive_words = ['great', 'amazing', 'excellent', 'love', 'perfect', 'awesome', 
                     'fantastic', 'wonderful', 'outstanding', 'brilliant']
    negative_words = ['terrible', 'awful', 'hate', 'worst', 'horrible', 'useless', 
                     'bug', 'crash', 'slow', 'disappointing']
    neutral_words = ['okay', 'average', 'normal', 'fine', 'decent', 'standard']
    
    def generate_realistic_review(rating):
        if rating >= 4:
            words = random.sample(positive_words, 2)
            templates = [
                f"This app is {words[0]}! Really {words[1]} experience.",
                f"Love this app, very {words[0]} and {words[1]}!",
                f"{words[0].title()} app with {words[1]} features!"
            ]
        elif rating <= 2:
            words = random.sample(negative_words, 2)
            templates = [
                f"This app is {words[0]}. Very {words[1]} experience.",
                f"Hate this app, full of {words[0]} and {words[1]} issues.",
                f"{words[0].title()} app with {words[1]} problems!"
            ]
        else:
            words = random.sample(neutral_words, 2)
            templates = [
                f"The app is {words[0]}, nothing special. Just {words[1]}.",
                f"Pretty {words[0]} app, {words[1]} overall.",
                f"{words[0].title()} performance, {words[1]} user experience."
            ]
        return random.choice(templates)
    
    # Generate dataset
    dataset = []
    for i in range(n_samples):
        rating = random.choices([1, 2, 3, 4, 5], weights=[8, 12, 20, 35, 25])[0]
        app_id = random.randint(1, 50)
        
        record = {
            'id': i + 1,
            'app_id': app_id,
            'app_name': f"App_{app_id}",
            'category': random.choice(app_categories),
            'rating': rating,
            'review_text': generate_realistic_review(rating),
            'helpful_votes': random.randint(0, 20),
            'total_votes': random.randint(1, 30),
            'app_size_mb': round(random.uniform(5, 500), 1),
            'price': random.choices([0, 0.99, 1.99, 2.99, 4.99], weights=[60, 20, 10, 5, 5])[0],
            'version': random.choice(['1.0', '1.1', '1.2', '2.0', '2.1']),
            'date': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
            'user_id': random.randint(1000, 9999)
        }
        
        # Calculate derived fields
        record['helpful_ratio'] = record['helpful_votes'] / record['total_votes']
        record['is_free'] = 1 if record['price'] == 0 else 0
        record['review_length'] = len(record['review_text'])
        record['review_word_count'] = len(record['review_text'].split())
        
        dataset.append(record)
    
    return pd.DataFrame(dataset)

def clean_text(text):
    """Simple text cleaning"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def simple_sentiment_analysis(text):
    """Simple rule-based sentiment analysis"""
    positive_words = ['great', 'amazing', 'excellent', 'love', 'perfect', 'awesome', 
                     'fantastic', 'wonderful', 'outstanding', 'brilliant']
    negative_words = ['terrible', 'awful', 'hate', 'worst', 'horrible', 'useless', 
                     'bug', 'crash', 'slow', 'disappointing']
    
    text = clean_text(text)
    words = text.split()
    
    pos_score = sum(1 for word in words if word in positive_words)
    neg_score = sum(1 for word in words if word in negative_words)
    
    if pos_score > neg_score:
        return 'Positive', pos_score - neg_score
    elif neg_score > pos_score:
        return 'Negative', neg_score - pos_score
    else:
        return 'Neutral', 0

def preprocess_data(df):
    """Preprocess the dataset"""
    # Apply text cleaning and sentiment analysis
    df['cleaned_review'] = df['review_text'].apply(clean_text)
    
    sentiment_data = df['review_text'].apply(simple_sentiment_analysis)
    df['sentiment_label'] = sentiment_data.apply(lambda x: x[0])
    df['sentiment_score'] = sentiment_data.apply(lambda x: x[1])
    
    # Categorize ratings
    df['rating_category'] = df['rating'].apply(
        lambda x: 'High' if x >= 4 else ('Low' if x <= 2 else 'Medium')
    )
    
    # Categorize app size
    df['size_category'] = df['app_size_mb'].apply(
        lambda x: 'Small' if x < 50 else ('Medium' if x < 150 else 'Large')
    )
    
    return df

# Main app
def main():
    st.markdown('<h1 class="main-header">📱 App Store Reviews Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Data generation
    if not st.session_state.data_generated:
        if st.sidebar.button("🚀 Generate Dataset", type="primary"):
            with st.spinner("Generating dataset..."):
                st.session_state.dataset = generate_dataset()
                st.session_state.dataset = preprocess_data(st.session_state.dataset)
                st.session_state.data_generated = True
            st.success("Dataset generated successfully!")
    
    if st.session_state.data_generated and st.session_state.dataset is not None:
        df = st.session_state.dataset
        
        # Dataset info
        st.sidebar.info(f"📊 **Dataset Info**\n\n"
                       f"• Total Reviews: {len(df):,}\n"
                       f"• Unique Apps: {df['app_id'].nunique()}\n"
                       f"• Categories: {df['category'].nunique()}\n"
                       f"• Date Range: {df['date'].min()} to {df['date'].max()}")
        
        # Filters
        st.sidebar.subheader("🔍 Filters")
        
        # Category filter
        categories = st.sidebar.multiselect(
            "Select Categories",
            options=df['category'].unique(),
            default=df['category'].unique()
        )
        
        # Rating filter
        rating_range = st.sidebar.slider(
            "Rating Range",
            min_value=1,
            max_value=5,
            value=(1, 5)
        )
        
        # Price filter
        price_filter = st.sidebar.selectbox(
            "Price Filter",
            options=["All", "Free Only", "Paid Only"]
        )
        
        # Apply filters
        filtered_df = df[
            (df['category'].isin(categories)) &
            (df['rating'] >= rating_range[0]) &
            (df['rating'] <= rating_range[1])
        ]
        
        if price_filter == "Free Only":
            filtered_df = filtered_df[filtered_df['is_free'] == 1]
        elif price_filter == "Paid Only":
            filtered_df = filtered_df[filtered_df['is_free'] == 0]
        
        # Main dashboard
        if len(filtered_df) > 0:
            # Key Metrics
            st.subheader("📈 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_rating = filtered_df['rating'].mean()
                st.metric("Average Rating", f"{avg_rating:.2f}", delta=f"{avg_rating - 3:.2f}")
            
            with col2:
                total_reviews = len(filtered_df)
                st.metric("Total Reviews", f"{total_reviews:,}")
            
            with col3:
                positive_pct = (filtered_df['sentiment_label'] == 'Positive').sum() / len(filtered_df) * 100
                st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            
            with col4:
                avg_helpful_ratio = filtered_df['helpful_ratio'].mean()
                st.metric("Avg Helpful Ratio", f"{avg_helpful_ratio:.2f}")
            
            # Charts
            st.subheader("📊 Analysis Charts")
            
            # Two columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Rating Distribution
                rating_counts = filtered_df['rating'].value_counts().sort_index()
                fig_rating = px.bar(
                    x=rating_counts.index,
                    y=rating_counts.values,
                    title="Rating Distribution",
                    labels={'x': 'Rating', 'y': 'Count'},
                    color=rating_counts.values,
                    color_continuous_scale='viridis'
                )
                fig_rating.update_layout(showlegend=False)
                st.plotly_chart(fig_rating, use_container_width=True)
                
                # Sentiment Analysis
                sentiment_counts = filtered_df['sentiment_label'].value_counts()
                fig_sentiment = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_map={
                        'Positive': '#2E8B57',
                        'Negative': '#DC143C',
                        'Neutral': '#FFA500'
                    }
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with col2:
                # Category Performance
                category_stats = filtered_df.groupby('category')['rating'].agg(['mean', 'count']).reset_index()
                category_stats = category_stats[category_stats['count'] >= 10]  # Filter categories with enough data
                
                fig_category = px.bar(
                    category_stats,
                    x='category',
                    y='mean',
                    title="Average Rating by Category",
                    labels={'mean': 'Average Rating', 'category': 'Category'},
                    color='mean',
                    color_continuous_scale='RdYlGn'
                )
                fig_category.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_category, use_container_width=True)
                
                # Price vs Rating
                fig_price = px.box(
                    filtered_df,
                    x='is_free',
                    y='rating',
                    title="Rating Distribution: Free vs Paid Apps",
                    labels={'is_free': 'App Type', 'rating': 'Rating'}
                )
                fig_price.update_xaxis(
                    tickvals=[0, 1],
                    ticktext=['Paid', 'Free']
                )
                st.plotly_chart(fig_price, use_container_width=True)
            
            # Correlation Heatmap
            st.subheader("🔗 Correlation Analysis")
            numeric_cols = ['rating', 'helpful_ratio', 'app_size_mb', 'price', 'review_word_count']
            corr_matrix = filtered_df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Text Analysis
            st.subheader("📝 Text Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Top words in positive reviews
                positive_reviews = filtered_df[filtered_df['sentiment_label'] == 'Positive']['cleaned_review']
                all_words = ' '.join(positive_reviews).split()
                word_counts = Counter([word for word in all_words if len(word) > 3])
                top_positive_words = dict(word_counts.most_common(10))
                
                fig_pos_words = px.bar(
                    x=list(top_positive_words.values()),
                    y=list(top_positive_words.keys()),
                    orientation='h',
                    title="Top Words in Positive Reviews",
                    labels={'x': 'Frequency', 'y': 'Words'}
                )
                st.plotly_chart(fig_pos_words, use_container_width=True)
            
            with col2:
                # Top words in negative reviews
                negative_reviews = filtered_df[filtered_df['sentiment_label'] == 'Negative']['cleaned_review']
                if len(negative_reviews) > 0:
                    all_words = ' '.join(negative_reviews).split()
                    word_counts = Counter([word for word in all_words if len(word) > 3])
                    top_negative_words = dict(word_counts.most_common(10))
                    
                    fig_neg_words = px.bar(
                        x=list(top_negative_words.values()),
                        y=list(top_negative_words.keys()),
                        orientation='h',
                        title="Top Words in Negative Reviews",
                        labels={'x': 'Frequency', 'y': 'Words'},
                        color_discrete_sequence=['#DC143C']
                    )
                    st.plotly_chart(fig_neg_words, use_container_width=True)
                else:
                    st.info("No negative reviews in filtered data")
            
            # Business Insights
            st.subheader("💡 Business Insights")
            
            # Generate insights
            insights = []
            
            # Best performing category
            best_category = category_stats.loc[category_stats['mean'].idxmax(), 'category']
            best_rating = category_stats['mean'].max()
            insights.append(f"🏆 **Best Category**: {best_category} with {best_rating:.2f} average rating")
            
            # Free vs Paid insight
            free_avg = filtered_df[filtered_df['is_free'] == 1]['rating'].mean()
            paid_avg = filtered_df[filtered_df['is_free'] == 0]['rating'].mean()
            if free_avg > paid_avg:
                insights.append(f"💰 **Pricing Insight**: Free apps perform better ({free_avg:.2f} vs {paid_avg:.2f})")
            else:
                insights.append(f"💰 **Pricing Insight**: Paid apps perform better ({paid_avg:.2f} vs {free_avg:.2f})")
            
            # Sentiment insight
            if positive_pct > 60:
                insights.append(f"😊 **Sentiment Health**: Good sentiment distribution ({positive_pct:.1f}% positive)")
            else:
                insights.append(f"😐 **Sentiment Health**: Room for improvement ({positive_pct:.1f}% positive)")
            
            # App size insight
            size_rating = filtered_df.groupby('size_category')['rating'].mean()
            best_size = size_rating.idxmax()
            insights.append(f"📱 **App Size**: {best_size} apps have highest ratings ({size_rating[best_size]:.2f})")
            
            for insight in insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("🎯 Recommendations")
            recommendations = [
                f"Focus development efforts on **{best_category}** category for best performance",
                "Monitor and respond quickly to negative reviews to improve sentiment",
                "Optimize app size based on category requirements",
                "Consider pricing strategy based on category and target audience",
                "Implement features that encourage positive user engagement"
            ]
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # Data Export
            st.subheader("📥 Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download Filtered Data (CSV)",
                    data=csv_data,
                    file_name=f"app_store_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create summary report
                summary_data = {
                    'total_reviews': len(filtered_df),
                    'average_rating': avg_rating,
                    'positive_sentiment_pct': positive_pct,
                    'best_category': best_category,
                    'insights': insights,
                    'recommendations': recommendations
                }
                
                import json
                json_data = json.dumps(summary_data, indent=2)
                st.download_button(
                    label="Download Summary Report (JSON)",
                    data=json_data,
                    file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            
            # Raw Data View
            with st.expander("🔍 View Raw Data"):
                st.dataframe(filtered_df.head(100), use_container_width=True)
        
        else:
            st.warning("No data matches the current filters. Please adjust your selection.")
    
    else:
        st.info("👆 Click 'Generate Dataset' in the sidebar to start the analysis!")
        
        # Show preview of what the app will do
        st.subheader("🎯 What This Dashboard Does")
        st.write("""
        This comprehensive big data analysis dashboard will:
        
        1. **Generate Realistic Data**: Creates 3,000 app store reviews with realistic patterns
        2. **Advanced Analytics**: Performs sentiment analysis, correlation analysis, and statistical insights  
        3. **Interactive Visualizations**: Dynamic charts and plots that update based on your filters
        4. **Business Intelligence**: Actionable insights and recommendations for app developers
        5. **Data Export**: Download processed data and analysis reports
        
        **Features Include:**
        - 📊 Rating and sentiment distribution analysis
        - 🏷️ Category performance comparison  
        - 💰 Free vs paid app analysis
        - 📝 Text mining and word frequency analysis
        - 🔗 Correlation analysis between app features
        - 💡 AI-generated business insights and recommendations
        """)

if __name__ == "__main__":
    main()
