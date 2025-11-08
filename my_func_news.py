
import my_func_support as sup
import my_func as f
import requests
import os
from datetime import datetime


api_key = os.getenv('eodhd')


# Function defined for capital structure indicators
def fetch_news(symbol=None, market='US', limit=5, period='0d', news_day=None, tag=None, mode='latest'):

    if not symbol and not tag:
        raise ValueError("At least 'symbol' or 'tag' must be provided.")

    # Construct ticker if symbol is provided
    ticker = f"{symbol}.{market}" if symbol else None

    if news_day:
        # Validate the 'news_day' format
        try:
            # Parse the date string (YYYY-MM-DD format)
            datetime.strptime(news_day, '%Y-%m-%d')
            # If the day is valid, set both 'from' and 'to' to that day
            from_date = news_day
            to_date = news_day
        except ValueError:
            raise ValueError("Invalid 'news_day' format. It should be in 'YYYY-MM-DD' format.")
    else:
        # If 'news_day' is not provided, determine 'from' and 'to' based on 'period'
        from_date, to_date = sup.parse_period(period)

    params = {}
    # Add the 'ticker' only if provided
    if ticker:
        params['s'] = ticker

    # Add the 'tag' only if provided
    if tag:
        params['t'] = tag

    # If mode is 'positive' or 'negative', set limit to 1000 when fetching from API
    fetch_limit = 1000 if mode in ['positive', 'negative'] else limit

    # Prepare parameters for the API call
    params.update({
        'offset': 0,
        'limit': fetch_limit,  # Use larger limit for filtering in 'positive'/'negative' modes
        'from': from_date,
        'to': to_date,
        'api_token': api_key,
    })

    # Prepare url
    endpoint = f.generate_url('news')

    # Make the API call
    response = requests.get(endpoint, params=params)

    # Check the status of the API response
    if response.status_code == 200:
        data = response.json()
    else:
        raise ValueError(f"News data could not be retrieved. Response status code: {response.status_code}")

    # Process the articles to extract required fields
    processed_articles = {}
    for article in data:
        sentiment = article.get('sentiment', {})
        # Use the article title as the key in the dictionary
        processed_articles[article['title']] = {
            'date': article.get('date'),
            'content': article.get('content'),
            'link': article.get('link'),
            'polarity': sentiment.get('polarity', 0)  # Store polarity separately for easy access
        }

    # If mode is 'positive', select top 'x' most bullish articles
    if mode == 'positive':
        sorted_articles = sorted(processed_articles.items(), key=lambda x: x[1]['polarity'], reverse=True)
        processed_articles = dict(sorted_articles[:limit])

    # If mode is 'negative', select top 'x' most bearish articles
    elif mode == 'negative':
        sorted_articles = sorted(processed_articles.items(), key=lambda x: x[1]['polarity'])
        processed_articles = dict(sorted_articles[:limit])

    # Get the count of news articles retrieved
    article_count = len(processed_articles)

    # Find the most positive and most negative articles using max() and min()
    most_positive_article = max(processed_articles.items(), key=lambda x: x[1]['polarity'], default=None)
    most_negative_article = min(processed_articles.items(), key=lambda x: x[1]['polarity'], default=None)

    # Extract the titles of the most positive and most negative articles
    most_positive_title = most_positive_article[0] if most_positive_article else None
    most_negative_title = most_negative_article[0] if most_negative_article else None

    # Return the processed articles, article count, and the most positive/negative titles
    return {
        'article_count': article_count,
        'most_positive_article': most_positive_title,
        'most_negative_article': most_negative_title,
        'articles': processed_articles
    }



