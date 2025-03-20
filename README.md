# Reddit Business Insights Scraper

A Streamlit application that scrapes and analyzes Reddit posts related to business ideas, with a focus on AI and app development.

## Features

- Search for relevant subreddits based on your query
- Fetch and analyze posts from selected subreddits
- Sentiment analysis of posts and comments
- Generate detailed reports with insights
- Export data in CSV format
- Interactive visualizations

## Setup

1. Clone the repository:
```bash
git clone https://github.com/SukinShetty/redditscraper.git
cd redditscraper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Reddit API credentials:
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
```

4. Run the application:
```bash
python -m streamlit run simple_streamlit_app.py
```

## Usage

1. Enter your search query (e.g., "gen ai app ideas")
2. Set the maximum number of subreddits and posts per subreddit
3. Click "Generate Report" to start the analysis
4. View the results in the Report, Visualizations, and Raw Data tabs
5. Download the analysis as a text report or CSV file

## Troubleshooting

### Reddit API Issues
If you encounter a 401 HTTP response:
1. Double-check your Reddit API credentials in the `.env` file
2. Make sure the client ID and secret are correct
3. Verify that your Reddit account has the necessary permissions

### NLTK Issues
If you encounter NLTK-related errors:
1. The application will automatically download required NLTK data
2. If you still encounter issues, manually download the required package:
```python
import nltk
nltk.download('punkt')
```

## Future Enhancements

- Add data visualizations (word clouds, trend graphs)
- Implement sentiment analysis
- Add competitor analysis features
- Create email reports for regular updates
- Add user authentication for the web interface

## License

MIT

## Disclaimer

This tool is for educational purposes only. Be sure to comply with Reddit's API Terms of Service and rate limits when using this scraper. 