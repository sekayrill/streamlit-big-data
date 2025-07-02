# PROYEK ANALISIS BIG DATA - APP STORE REVIEWS (VERSI SEDERHANA)
# Versi ini menggunakan library minimal yang biasanya sudah tersedia

import os
import json
import csv
import random
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import re

print("=== PROYEK ANALISIS BIG DATA: APP STORE REVIEWS ===")
print("Analisis Sentiment dan Performa Aplikasi Mobile (Versi Sederhana)\n")

# ====================================================================
# 1. PENGUMPULAN DATA (DATA GENERATION)
# ====================================================================
print("1. PENGUMPULAN DATA")
print("-" * 50)

# Simulasi data yang realistis
random.seed(42)

# Master data
app_categories = ['Games', 'Social', 'Productivity', 'Entertainment', 'Business', 
                 'Education', 'Health & Fitness', 'Finance', 'News', 'Travel']
app_names = [f"App_{i}" for i in range(1, 51)]

# Positive dan negative words untuk sentiment
positive_words = ['great', 'amazing', 'excellent', 'love', 'perfect', 'awesome', 
                 'fantastic', 'wonderful', 'outstanding', 'brilliant']
negative_words = ['terrible', 'awful', 'hate', 'worst', 'horrible', 'useless', 
                 'bug', 'crash', 'slow', 'disappointing']
neutral_words = ['okay', 'average', 'normal', 'fine', 'decent', 'standard']

def generate_realistic_review(rating):
    """Generate realistic review text based on rating"""
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
n_samples = 3000
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

print(f"Dataset berhasil dibuat: {len(dataset)} records")
print(f"Data Terstruktur: rating, helpful_votes, app_size_mb, price, dll.")
print(f"Data Tidak Terstruktur: review_text")

# Sample data preview
print("\nSample data:")
for i in range(3):
    record = dataset[i]
    print(f"  {record['app_name']} | Rating: {record['rating']} | Review: {record['review_text'][:50]}...")

# ====================================================================
# 2. DATA PREPROCESSING
# ====================================================================
print("\n\n2. DATA PREPROCESSING")
print("-" * 50)

def clean_text(text):
    """Simple text cleaning"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def simple_sentiment_analysis(text):
    """Simple rule-based sentiment analysis"""
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

# Apply preprocessing
for record in dataset:
    record['cleaned_review'] = clean_text(record['review_text'])
    sentiment_label, sentiment_score = simple_sentiment_analysis(record['review_text'])
    record['sentiment_label'] = sentiment_label
    record['sentiment_score'] = sentiment_score
    
    # Categorize rating
    if record['rating'] >= 4:
        record['rating_category'] = 'High'
    elif record['rating'] <= 2:
        record['rating_category'] = 'Low'
    else:
        record['rating_category'] = 'Medium'
    
    # Categorize app size
    if record['app_size_mb'] < 50:
        record['size_category'] = 'Small'
    elif record['app_size_mb'] < 150:
        record['size_category'] = 'Medium'
    else:
        record['size_category'] = 'Large'

print("✓ Data preprocessing selesai")

# Basic sentiment statistics
sentiment_counts = Counter(record['sentiment_label'] for record in dataset)
print(f"Sentiment distribution: {dict(sentiment_counts)}")

# ====================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ====================================================================
print("\n\n3. EXPLORATORY DATA ANALYSIS (EDA)")
print("-" * 50)

def calculate_statistics(data, field):
    """Calculate basic statistics for a numeric field"""
    values = [record[field] for record in data]
    values.sort()
    n = len(values)
    
    stats = {
        'count': n,
        'mean': sum(values) / n,
        'median': values[n//2] if n % 2 == 1 else (values[n//2-1] + values[n//2]) / 2,
        'min': min(values),
        'max': max(values),
        'std': math.sqrt(sum((x - sum(values)/n)**2 for x in values) / n)
    }
    return stats

# Basic statistics
print("STATISTIK DASAR:")
print("-" * 30)

# Rating statistics
rating_stats = calculate_statistics(dataset, 'rating')
print(f"Rating - Mean: {rating_stats['mean']:.2f}, Std: {rating_stats['std']:.2f}")

# Rating distribution
rating_dist = Counter(record['rating'] for record in dataset)
print("Rating Distribution:")
for rating in sorted(rating_dist.keys()):
    percentage = rating_dist[rating] / len(dataset) * 100
    print(f"  {rating} stars: {rating_dist[rating]} ({percentage:.1f}%)")

# Category analysis
category_stats = defaultdict(list)
for record in dataset:
    category_stats[record['category']].append(record['rating'])

print("\nRating by Category:")
for category in sorted(category_stats.keys()):
    ratings = category_stats[category]
    avg_rating = sum(ratings) / len(ratings)
    print(f"  {category}: {avg_rating:.2f} ({len(ratings)} reviews)")

# Sentiment analysis
print(f"\nSentiment Analysis:")
for sentiment in ['Positive', 'Neutral', 'Negative']:
    count = sentiment_counts[sentiment]
    percentage = count / len(dataset) * 100
    print(f"  {sentiment}: {count} ({percentage:.1f}%)")

# ====================================================================
# 4. ANALISIS KORELASI
# ====================================================================
print("\n\n4. ANALISIS KORELASI")
print("-" * 50)

def calculate_correlation(data, field1, field2):
    """Calculate correlation between two numeric fields"""
    values1 = [record[field1] for record in data]
    values2 = [record[field2] for record in data]
    
    n = len(values1)
    mean1 = sum(values1) / n
    mean2 = sum(values2) / n
    
    numerator = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(n))
    denominator = math.sqrt(sum((x - mean1)**2 for x in values1) * sum((x - mean2)**2 for x in values2))
    
    return numerator / denominator if denominator != 0 else 0

# Calculate key correlations
correlations = {
    'Rating vs App Size': calculate_correlation(dataset, 'rating', 'app_size_mb'),
    'Rating vs Price': calculate_correlation(dataset, 'rating', 'price'),
    'Rating vs Helpful Ratio': calculate_correlation(dataset, 'rating', 'helpful_ratio'),
    'App Size vs Price': calculate_correlation(dataset, 'app_size_mb', 'price')
}

print("KORELASI ANTAR VARIABEL:")
for relationship, corr in correlations.items():
    print(f"  {relationship}: {corr:.3f}")

# ====================================================================
# 5. MACHINE LEARNING SEDERHANA
# ====================================================================
print("\n\n5. MACHINE LEARNING ANALYSIS")
print("-" * 50)

def simple_prediction_model(data):
    """Simple rule-based prediction model"""
    correct_predictions = 0
    total_predictions = len(data)
    
    for record in data:
        # Simple rule: predict high rating if positive sentiment and good helpful ratio
        predicted_high_rating = (
            record['sentiment_label'] == 'Positive' and 
            record['helpful_ratio'] > 0.5
        )
        actual_high_rating = record['rating'] >= 4
        
        if predicted_high_rating == actual_high_rating:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy

# Test prediction model
accuracy = simple_prediction_model(dataset)
print(f"Simple Prediction Model Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

# Feature importance analysis
def analyze_feature_importance(data):
    """Analyze which features are most important for high ratings"""
    high_rating_apps = [r for r in data if r['rating'] >= 4]
    low_rating_apps = [r for r in data if r['rating'] <= 2]
    
    print("FEATURE ANALYSIS:")
    print("-" * 20)
    
    # Sentiment distribution
    high_sentiment = Counter(r['sentiment_label'] for r in high_rating_apps)
    low_sentiment = Counter(r['sentiment_label'] for r in low_rating_apps)
    
    print("High Rating Apps Sentiment:")
    for sentiment, count in high_sentiment.items():
        pct = count / len(high_rating_apps) * 100
        print(f"  {sentiment}: {pct:.1f}%")
    
    print("Low Rating Apps Sentiment:")
    for sentiment, count in low_sentiment.items():
        pct = count / len(low_rating_apps) * 100
        print(f"  {sentiment}: {pct:.1f}%")
    
    # Price analysis
    high_free_pct = sum(1 for r in high_rating_apps if r['is_free']) / len(high_rating_apps) * 100
    low_free_pct = sum(1 for r in low_rating_apps if r['is_free']) / len(low_rating_apps) * 100
    
    print(f"\nFree Apps:")
    print(f"  High Rating: {high_free_pct:.1f}%")
    print(f"  Low Rating: {low_free_pct:.1f}%")

analyze_feature_importance(dataset)

# ====================================================================
# 6. TEXT ANALYSIS
# ====================================================================
print("\n\n6. TEXT ANALYSIS")
print("-" * 50)

def analyze_text_patterns(data):
    """Analyze text patterns in reviews"""
    positive_reviews = [r['cleaned_review'] for r in data if r['sentiment_label'] == 'Positive']
    negative_reviews = [r['cleaned_review'] for r in data if r['sentiment_label'] == 'Negative']
    
    # Word frequency analysis
    positive_words_count = Counter()
    negative_words_count = Counter()
    
    for review in positive_reviews:
        words = review.split()
        positive_words_count.update(words)
    
    for review in negative_reviews:
        words = review.split()
        negative_words_count.update(words)
    
    print("TOP WORDS IN POSITIVE REVIEWS:")
    for word, count in positive_words_count.most_common(10):
        if len(word) > 2:  # Skip short words
            print(f"  {word}: {count}")
    
    print("\nTOP WORDS IN NEGATIVE REVIEWS:")
    for word, count in negative_words_count.most_common(10):
        if len(word) > 2:  # Skip short words
            print(f"  {word}: {count}")
    
    # Review length analysis
    avg_positive_length = sum(r['review_word_count'] for r in data if r['sentiment_label'] == 'Positive') / len([r for r in data if r['sentiment_label'] == 'Positive'])
    avg_negative_length = sum(r['review_word_count'] for r in data if r['sentiment_label'] == 'Negative') / len([r for r in data if r['sentiment_label'] == 'Negative'])
    
    print(f"\nREVIEW LENGTH ANALYSIS:")
    print(f"  Positive reviews avg length: {avg_positive_length:.1f} words")
    print(f"  Negative reviews avg length: {avg_negative_length:.1f} words")

analyze_text_patterns(dataset)

# ====================================================================
# 7. BUSINESS INSIGHTS
# ====================================================================
print("\n\n7. BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

def generate_business_insights(data):
    """Generate business insights from analysis"""
    insights = []
    recommendations = []
    
    # Basic metrics
    total_apps = len(set(r['app_id'] for r in data))
    avg_rating = sum(r['rating'] for r in data) / len(data)
    high_rating_pct = sum(1 for r in data if r['rating'] >= 4) / len(data) * 100
    
    insights.append(f"Total {total_apps} apps analyzed with {len(data)} reviews")
    insights.append(f"Average rating: {avg_rating:.2f}/5.0")
    insights.append(f"{high_rating_pct:.1f}% of reviews are high-rated (4-5 stars)")
    
    # Category insights
    category_ratings = defaultdict(list)
    for record in data:
        category_ratings[record['category']].append(record['rating'])
    
    best_category = max(category_ratings, key=lambda x: sum(category_ratings[x])/len(category_ratings[x]))
    best_rating = sum(category_ratings[best_category]) / len(category_ratings[best_category])
    
    insights.append(f"Best performing category: {best_category} ({best_rating:.2f} avg rating)")
    
    # Sentiment insights
    positive_pct = sum(1 for r in data if r['sentiment_label'] == 'Positive') / len(data) * 100
    insights.append(f"{positive_pct:.1f}% of reviews have positive sentiment")
    
    # Price insights
    free_apps = [r for r in data if r['is_free'] == 1]
    paid_apps = [r for r in data if r['is_free'] == 0]
    
    free_avg_rating = sum(r['rating'] for r in free_apps) / len(free_apps)
    paid_avg_rating = sum(r['rating'] for r in paid_apps) / len(paid_apps)
    
    insights.append(f"Free apps avg rating: {free_avg_rating:.2f}")
    insights.append(f"Paid apps avg rating: {paid_avg_rating:.2f}")
    
    # Generate recommendations
    recommendations.append("Focus on positive sentiment features to improve ratings")
    recommendations.append(f"Prioritize development in {best_category} category")
    recommendations.append("Monitor and respond to negative feedback quickly")
    recommendations.append("Optimize app size for better user experience")
    recommendations.append("Consider pricing strategy based on category and features")
    
    return insights, recommendations

insights, recommendations = generate_business_insights(dataset)

print("KEY INSIGHTS:")
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")

print("\nBUSINESS RECOMMENDATIONS:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

# ====================================================================
# 8. EXPORT RESULTS
# ====================================================================
print("\n\n8. EXPORT HASIL ANALISIS")
print("-" * 50)

# Export to CSV
csv_filename = 'app_store_analysis_results.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as file:
    if dataset:
        fieldnames = dataset[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

print(f"✓ Data exported to: {csv_filename}")

# Export insights to text file
insights_filename = 'business_insights.txt'
with open(insights_filename, 'w', encoding='utf-8') as file:
    file.write("=== APP STORE ANALYSIS - BUSINESS INSIGHTS ===\n\n")
    file.write("KEY INSIGHTS:\n")
    for i, insight in enumerate(insights, 1):
        file.write(f"{i}. {insight}\n")
    file.write("\nBUSINESS RECOMMENDATIONS:\n")
    for i, rec in enumerate(recommendations, 1):
        file.write(f"{i}. {rec}\n")
    
    file.write(f"\nANALYSIS SUMMARY:\n")
    file.write(f"- Dataset size: {len(dataset)} records\n")
    file.write(f"- Categories analyzed: {len(set(r['category'] for r in dataset))}\n")
    file.write(f"- Apps analyzed: {len(set(r['app_id'] for r in dataset))}\n")
    file.write(f"- Sentiment distribution: {dict(sentiment_counts)}\n")

print(f"✓ Insights exported to: {insights_filename}")

# Export summary statistics
stats_filename = 'analysis_statistics.json'
summary_stats = {
    'dataset_size': len(dataset),
    'rating_distribution': dict(Counter(r['rating'] for r in dataset)),
    'sentiment_distribution': dict(sentiment_counts),
    'category_distribution': dict(Counter(r['category'] for r in dataset)),
    'correlations': correlations,
    'model_accuracy': accuracy
}

with open(stats_filename, 'w') as file:
    json.dump(summary_stats, file, indent=2)

print(f"✓ Statistics exported to: {stats_filename}")

print("\n" + "="*60)
print("🎉 PROYEK ANALISIS BIG DATA SELESAI!")
print("="*60)
print("Files yang dihasilkan:")
print(f"• {csv_filename} - Dataset hasil preprocessing")
print(f"• {insights_filename} - Business insights dan recommendations")
print(f"• {stats_filename} - Summary statistics")
print("\nProyek ini mencakup:")
print("✓ Data collection (structured + unstructured)")
print("✓ Data preprocessing dan cleaning")
print("✓ Exploratory Data Analysis (EDA)")
print("✓ Text analysis dan sentiment analysis")
print("✓ Machine learning prediction")
print("✓ Business insights generation")
print("✓ Data export dalam multiple formats")
print("\nSiap untuk presentasi dan dashboard development!")