import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import re
from urllib.parse import urljoin, quote_plus
import os
import json
import logging
from datetime import datetime
import numpy as np
from fake_useragent import UserAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class amazonScraper:
    def __init__(self, base_url="https://www.amazon.com"):
        self.base_url = base_url
        self.user_agent = UserAgent()
        self.products = []
        self.session = requests.Session()
        self.delay_range = (3, 7)  # Random delay between requests

    def get_headers(self):
        """Generate random user agent headers to avoid detection"""
        return {
            "authority": self.base_url.split('//')[1],
            "pragma": "no-cache",
            "cache-control": "no-cache",
            "dnt": "1",
            "upgrade-insecure-requests": "1",
            "user-agent": self.user_agent.random,
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "sec-fetch-site": "none",
            "sec-fetch-mode": "navigate",
            "sec-fetch-dest": "document",
            "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        }

    def search_products(self, query, max_pages=3, alternative_urls=None):
        """Search for products on amazon using the provided query"""
        logger.info(f"Searching for: {query}")
        self.products = []

        # First try standard search pages
        products_found = self._try_search_pages(query, max_pages)

        # If no products found or very few, try alternative URLs
        if len(self.products) < 10:
            logger.info("Standard search failed. Trying alternative URLs...")
            alternative_urls = alternative_urls or [
                f"{self.base_url}/s?k={quote_plus(query)}&ref=nb_sb_noss",
                f"{self.base_url}/s?k={quote_plus(query)}&crid=2M096C61O4MLT&qid=1653308124",
                f"{self.base_url}/s?k={quote_plus(query)}&i=hpc",
                f"{self.base_url}/s?k={quote_plus(query)}&i=beauty"
            ]

            for alt_url in alternative_urls:
                logger.info(f"Trying alternative URL: {alt_url}")
                self._try_alternative_url(alt_url)
                # If we found enough products, break
                if len(self.products) >= 20:
                    break

        logger.info(f"Extracted {len(self.products)} products in total")
        return self.products

    def _try_search_pages(self, query, max_pages):
        """Try scraping standard search result pages"""
        encoded_query = quote_plus(query)
        for page in range(1, max_pages + 1):
            url = f"{self.base_url}/s?k={encoded_query}&page={page}"
            logger.info(f"Scraping page {page}: {url}")

            soup = self._make_request(url)
            if not soup:
                continue

            # Extract both sponsored and regular products
            sponsored_products = self._extract_sponsored_products(soup)
            regular_products = self._extract_products(soup)

            products_on_page = len(sponsored_products) + len(regular_products)
            logger.info(f"Extracted {products_on_page} products from this page")

            # If no products found on this page, no need to continue
            if products_on_page == 0 and page > 1:
                break

            # Add delay before next request
            if page < max_pages:
                delay = random.uniform(*self.delay_range)
                logger.info(f"Waiting {delay:.2f} seconds before next request...")
                time.sleep(delay)

    def _try_alternative_url(self, url):
        """Try scraping from an alternative URL"""
        soup = self._make_request(url)
        if not soup:
            return

        # Extract products
        sponsored_products = self._extract_sponsored_products(soup)
        regular_products = self._extract_products(soup)

        # Add delay before next request
        delay = random.uniform(*self.delay_range)
        logger.info(f"Waiting {delay:.2f} seconds before next request...")
        time.sleep(delay)

    def _make_request(self, url):
        """Make HTTP request with error handling and retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = self.get_headers()
                response = self.session.get(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    return BeautifulSoup(response.content, "html.parser")
                elif response.status_code == 503 or response.status_code == 429:
                    logger.warning(f"Rate limited (Status: {response.status_code}). Waiting longer...")
                    time.sleep(20 + random.uniform(5, 15))
                else:
                    logger.warning(f"Failed to fetch URL: {url} with status code: {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")

            # Increase delay on retry
            time.sleep(random.uniform(5, 10))

        return None

    def _extract_sponsored_products(self, soup):
        """Extract sponsored products from the page"""
        sponsored_products = []

        # Multiple potential selectors for sponsored products
        sponsored_elements = soup.select('div.s-result-item[data-component-type="sp-sponsored-result"]')

        for idx, product in enumerate(sponsored_elements):
            try:
                product_data = self._parse_product_element(product, is_sponsored=True)
                if product_data:
                    sponsored_products.append(product_data)
                    self.products.append(product_data)
                    logger.info(f"Found sponsored product: {product_data.get('title', 'Unknown')[:50]}")
            except Exception as e:
                logger.error(f"Error extracting sponsored product {idx}: {e}")

        return sponsored_products

    def _extract_products(self, soup):
        """Extract regular (non-sponsored) products from the page"""
        regular_products = []

        # Look for standard product elements
        product_elements = soup.select('div.s-result-item[data-component-type="s-search-result"]')

        for idx, product in enumerate(product_elements):
            try:
                # Skip sponsored products that might be included
                if product.select_one('.s-label-popover-default') or 'Sponsored' in product.text:
                    continue

                product_data = self._parse_product_element(product, is_sponsored=True)
                if product_data:
                    regular_products.append(product_data)
                    self.products.append(product_data)
                    logger.info(f"Found product: {product_data.get('title', 'Unknown')[:50]}")
            except Exception as e:
                logger.error(f"Error extracting regular product {idx}: {e}")

        # If few results, try alternative selectors
        if len(regular_products) < 3:
            try:
                alternative_products = soup.select('div.sg-col-4-of-12')
                for product in alternative_products:
                    if product.select_one('.s-label-popover-default') or 'Sponsored' in product.text:
                        continue

                    product_data = self._parse_product_element(product, is_sponsored=True)
                    if product_data:
                        regular_products.append(product_data)
                        self.products.append(product_data)
            except Exception as e:
                logger.error(f"Error extracting products with alternative selector: {e}")

        return regular_products

    def _parse_product_element(self, product_element, is_sponsored=True):
        """Parse a product element to extract details"""
        try:
            # Find product title
            title_element = product_element.select_one('h2 a.a-link-normal span') or product_element.select_one('h2 a span.a-text-normal')
            if not title_element:
                title_element = product_element.select_one('.a-size-base-plus') or product_element.select_one('.a-size-medium')

            title = title_element.text.strip() if title_element else "Unknown Title"

            # Find product link
            link_element = product_element.select_one('h2 a.a-link-normal') or product_element.select_one('a.a-link-normal.s-no-outline')
            link = urljoin(self.base_url, link_element['href']) if link_element and 'href' in link_element.attrs else None

            # Find product image
            img_element = product_element.select_one('img.s-image')
            image_url = img_element['src'] if img_element and 'src' in img_element.attrs else None

            # Find product price
            price_element = product_element.select_one('.a-price .a-offscreen') or product_element.select_one('.a-price-whole')
            price = price_element.text.strip() if price_element else "Price not available"

            # Find product ratings
            rating_element = product_element.select_one('span[aria-label*="stars"]') or product_element.select_one('i.a-icon-star-small')
            rating = None
            if rating_element:
                rating_text = rating_element.get('aria-label', '') or rating_element.text
                rating_match = re.search(r'(\d+\.\d+|\d+)', rating_text)
                rating = float(rating_match.group(1)) if rating_match else None

            # Find number of reviews
            reviews_element = product_element.select_one('span.a-size-base') or product_element.select_one('.a-link-normal .a-size-base')
            reviews = None
            if reviews_element:
                reviews_text = reviews_element.text.strip()
                reviews_match = re.search(r'(\d+(?:,\d+)*)', reviews_text)
                reviews = reviews_match.group(1).replace(',', '') if reviews_match else "0"
                reviews = int(reviews) if reviews.isdigit() else 0

            # Set brand as the first word of the title
            brand = title.split()[0] if title != "Unknown Title" else "Unknown"

            # Find if product is amazon's Choice or Best Seller
            badges = []
            amazon_choice = product_element.select_one('span.a-badge-label')
            if amazon_choice and "amazon's Choice" in amazon_choice.text:
                badges.append("amazon's Choice")
            bestseller = product_element.select_one('span.a-badge-text')
            if bestseller and "Best Seller" in bestseller.text:
                badges.append("Best Seller")

            # Save ASIN (amazon Standard Identification Number)
            asin = None
            if link:
                asin_match = re.search(r'/dp/([A-Z0-9]{10})/', link)
                if asin_match:
                    asin = asin_match.group(1)

            # Create product dictionary
            product_data = {
                'title': title,
                'url': link,
                'image_url': image_url,
                'price': price,
                'rating': rating,
                'reviews': reviews,
                'brand': brand,
                'badges': ','.join(badges) if badges else None,
                'asin': asin,
                'is_sponsored': is_sponsored,
                'scrape_date': datetime.now().strftime('%Y-%m-%d')
            }

            return product_data
        except Exception as e:
            logger.error(f"Error parsing product element: {e}")
            return None

    def scrape_product_reviews(self, url_or_asin, max_pages=3):
        """Scrape reviews for a specific product"""
        asin = url_or_asin
        if not asin.isalnum() or len(asin) != 10:
            # Try to extract ASIN from URL
            asin_match = re.search(r'/dp/([A-Z0-9]{10})/', url_or_asin)
            if asin_match:
                asin = asin_match.group(1)
            else:
                logger.error(f"Invalid ASIN or URL: {url_or_asin}")
                return []

        logger.info(f"Scraping reviews for product ASIN: {asin}")
        reviews = []

        for page in range(1, max_pages + 1):
            review_url = f"{self.base_url}/product-reviews/{asin}/ref=cm_cr_arp_d_paging_btm_next_{page}?ie=UTF8&reviewerType=all_reviews&pageNumber={page}"
            logger.info(f"Scraping review page {page}: {review_url}")

            soup = self._make_request(review_url)
            if not soup:
                continue

            review_elements = soup.find_all("div", {"data-hook": "review"})
            if not review_elements:
                review_elements = soup.find_all("div", {"class": "a-section celwidget"})

            if not review_elements:
                logger.warning(f"No reviews found on page {page}")
                break

            for review in review_elements:
                try:
                    review_data = self._parse_review_element(review, asin)
                    if review_data:
                        reviews.append(review_data)
                except Exception as e:
                    logger.error(f"Error parsing review: {e}")

            logger.info(f"Extracted {len(review_elements)} reviews from page {page}")

            # Check if there are more pages
            next_page = soup.select_one('li.a-last a')
            if not next_page:
                logger.info("No more review pages available")
                break

            # Add delay before next request
            if page < max_pages:
                delay = random.uniform(*self.delay_range)
                logger.info(f"Waiting {delay:.2f} seconds before next request...")
                time.sleep(delay)

        logger.info(f"Total reviews scraped: {len(reviews)}")
        return reviews

    def _parse_review_element(self, review_element, asin):
        """Parse a single review element"""
        try:
            # Review title
            title_element = review_element.find("a", {"data-hook": "review-title"}) or review_element.find("span", {"class": "review-title"})
            title = title_element.get_text().strip() if title_element else "No Title"

            # Rating
            rating_element = review_element.find("i", {"data-hook": "review-star-rating"}) or review_element.find("span", {"class": "a-icon-alt"})
            rating = None
            if rating_element:
                rating_text = rating_element.get_text().strip()
                rating_match = re.search(r'(\d+\.\d+|\d+)', rating_text)
                rating = float(rating_match.group(1)) if rating_match else None

            # Review date
            date_element = review_element.find("span", {"data-hook": "review-date"}) or review_element.find("span", {"class": "review-date"})
            date = date_element.get_text().strip() if date_element else None

            # Review text
            text_element = review_element.find("span", {"data-hook": "review-body"}) or review_element.find("span", {"class": "review-text"})
            if not text_element:
                text_element = review_element.find("div", {"class": "a-row review-data"})
            text = text_element.get_text().strip() if text_element else "No review text"

            # Verified purchase
            verified = False
            verified_element = review_element.find("span", {"data-hook": "avp-badge"})
            if verified_element and "Verified Purchase" in verified_element.get_text():
                verified = True

            # Helpful votes
            helpful_element = review_element.find("span", {"data-hook": "helpful-vote-statement"})
            helpful_votes = 0
            if helpful_element:
                votes_match = re.search(r'(\d+)', helpful_element.get_text())
                helpful_votes = int(votes_match.group(1)) if votes_match else 0

            # Reviewer name
            reviewer_element = review_element.find("span", {"class": "a-profile-name"})
            reviewer = reviewer_element.get_text().strip() if reviewer_element else "Anonymous"

            return {
                'asin': asin,
                'reviewer': reviewer,
                'title': title,
                'rating': rating,
                'date': date,
                'text': text,
                'verified_purchase': verified,
                'helpful_votes': helpful_votes
            }
        except Exception as e:
            logger.error(f"Error parsing review element: {e}")
            return None

    def load_sample_data(self):
        """Load sample data for testing"""
        logger.info("Loading sample data")
        sample_data = [
            {
                'title': 'Sample Product 1',
                'url': 'https://www.amazon.in/sample1',
                'image_url': 'https://example.com/img1.jpg',
                'price': '₹1,999.00',
                'rating': 4.5,
                'reviews': 120,
                'brand': 'Brand A',
                'badges': "amazon's Choice",
                'asin': 'B0123456789',
                'is_sponsored': True,
                'scrape_date': datetime.now().strftime('%Y-%m-%d')
            },
            {
                'title': 'Sample Product 2',
                'url': 'https://www.amazon.in/sample2',
                'image_url': 'https://example.com/img2.jpg',
                'price': '₹2,499.00',
                'rating': 3.8,
                'reviews': 85,
                'brand': 'Brand B',
                'badges': None,
                'asin': 'B0123456790',
                'is_sponsored': True,
                'scrape_date': datetime.now().strftime('%Y-%m-%d')
            }
        ]
        self.products = sample_data
        return sample_data

    def save_to_csv(self, filename="raw_amazon_products.csv"):
        """Save scraped products to CSV file"""
        if not self.products:
            logger.warning("No products to save")
            return None

        df = pd.DataFrame(self.products)
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"Data saved to {filename}")
        return filename

class DataCleaner:
    def __init__(self, data):
        """Initialize with either DataFrame or CSV file path"""
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, str) and os.path.exists(data):
            self.data = pd.read_csv(data)
        else:
            raise ValueError("Invalid data source. Please provide a DataFrame or valid CSV file path")

        logger.info(f"Loaded data with {len(self.data)} records")

    def clean_data(self):
        """Clean the dataset"""
        # Make a copy to avoid modifying the original
        df = self.data.copy()
        
        # Remove rows with "Unknown Title"
        initial_count = len(df)
        df = df[df['title'] != "Unknown Title"]
        unknown_title_count = initial_count - len(df)
        logger.info(f"Removed {unknown_title_count} rows with Unknown Title")

        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['title', 'price'], keep='first')
        duplicate_count = initial_count - len(df)
        logger.info(f"Removed {duplicate_count} duplicate entries")

        # Clean price
        if 'price' in df.columns:
            df['price'] = df['price'].apply(self._clean_price)

        # Clean reviews
        if 'reviews' in df.columns:
            df['reviews'] = df['reviews'].apply(self._clean_reviews)

        # Convert rating to float
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

        # Extract brand from title if missing
        if 'brand' in df.columns and 'title' in df.columns:
            df['brand'] = df.apply(lambda row: row['brand'] if pd.notna(row['brand']) and row['brand'] != 'Unknown'
                                   else row['title'].split()[0] if pd.notna(row['title']) else 'Unknown', axis=1)

        # Format date
        if 'scrape_date' in df.columns:
            df['scrape_date'] = pd.to_datetime(df['scrape_date'], errors='coerce').dt.strftime('%Y-%m-%d')

        # Fill missing values
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna('Unknown')
            elif df[col].dtype == float:
                df[col] = df[col].fillna(0)

        self.cleaned_data = df
        logger.info(f"Data cleaning complete. Final shape: {df.shape}")
        return df

    def _clean_price(self, price):
        """Clean price values"""
        if pd.isna(price) or price == "Price not available":
            return np.nan

        try:
            # Extract digits and decimal point
            price_match = re.search(r'[\d,]+\.\d+|\d+,\d+|\d+', str(price))
            if price_match:
                # Remove commas and convert to float
                return float(price_match.group(0).replace(',', ''))
            return np.nan
        except:
            return np.nan

    def _clean_reviews(self, reviews):
        """Clean review count values"""
        if pd.isna(reviews):
            return 0

        try:
            if isinstance(reviews, (int, float)):
                return int(reviews)

            # Extract digits
            reviews_match = re.search(r'(\d+(?:,\d+)*)', str(reviews))
            if reviews_match:
                # Remove commas and convert to int
                return int(reviews_match.group(1).replace(',', ''))
            return 0
        except:
            return 0

    def save_to_csv(self, filename="cleaned_amazon_data.csv"):
        """Save cleaned data to CSV"""
        if not hasattr(self, 'cleaned_data'):
            logger.warning("No cleaned data available. Run clean_data() first.")
            return None

        self.cleaned_data.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"Cleaned data saved to {filename}")
        return filename

class DataAnalyzer:
    def __init__(self, data):
        """Initialize with either DataFrame or CSV file path"""
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, str) and os.path.exists(data):
            self.data = pd.read_csv(data)
        else:
            raise ValueError("Invalid data source. Please provide a DataFrame or valid CSV file path")

        logger.info(f"Loaded data for analysis with {len(self.data)} records")

        # Configure visualization style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("viridis")

    def analyze_brand_performance(self):
        """Analyze brand performance metrics"""
        logger.info("Performing brand performance analysis")

        # Check if brand column exists
        if 'brand' not in self.data.columns:
            logger.error("Brand column not found in the dataset")
            return

        # Get brand frequencies
        brand_counts = self.data['brand'].value_counts()
        top_brands = brand_counts.head(10)

        print("\n=== Brand Performance Analysis ===")
        print("Top 5 Brands by Frequency:")
        print(top_brands.head())

        # Calculate average ratings by brand
        if 'rating' in self.data.columns:
            avg_rating_by_brand = self.data.groupby('brand')['rating'].mean().sort_values(ascending=False)
            top_rated_brands = avg_rating_by_brand.head(10)

            print("\nTop 5 Brands by Average Rating:")
            print(top_rated_brands.head())

            # Visualize brand performance
            self._visualize_brand_performance(top_brands, avg_rating_by_brand)

        print("\nAnalysis complete!")
        return top_brands, avg_rating_by_brand if 'rating' in self.data.columns else None

    def _visualize_brand_performance(self, top_brands, avg_rating_by_brand):
        """Visualize brand performance metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot brand frequency
        top_brands = top_brands.head(10)
        top_brands.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Top 10 Brands by Product Count')
        ax1.set_xlabel('Brand')
        ax1.set_ylabel('Number of Products')
        ax1.tick_params(axis='x', rotation=45)

        # Plot average rating by brand
        brand_ratings = avg_rating_by_brand.reindex(top_brands.index)
        brand_ratings.plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title('Average Rating of Top Brands')
        ax2.set_xlabel('Brand')
        ax2.set_ylabel('Average Rating')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 5)

        plt.tight_layout()
        plt.savefig('brand_performance.png')
        plt.close()

    def analyze_price_vs_rating(self):
        """Analyze relationship between price and rating"""
        logger.info("Performing price vs. rating analysis")

        # Check if required columns exist
        if 'price' not in self.data.columns or 'rating' not in self.data.columns:
            logger.error("Price or rating column not found in the dataset")
            return

        # Filter out missing values
        filtered_data = self.data[self.data['price'].notna() & self.data['rating'].notna()]

        # Calculate average price by rating
        avg_price_by_rating = filtered_data.groupby('rating')['price'].mean().reset_index()

        print("\n=== Price vs. Rating Analysis ===")
        print("Average Price by Rating:")
        print(avg_price_by_rating)

        # Calculate correlation
        correlation = filtered_data['price'].corr(filtered_data['rating'])
        print(f"\nCorrelation between Price and Rating: {correlation:.3f}")

        # Calculate price ranges by rating
        price_stats_by_rating = filtered_data.groupby('rating')['price'].agg(['mean', 'median', 'min', 'max'])
        print("\nPrice Statistics by Rating:")
        print(price_stats_by_rating)

        # Visualize price vs. rating
        self._visualize_price_vs_rating(filtered_data, avg_price_by_rating)

        print("\nAnalysis complete!")
        return avg_price_by_rating, correlation

    def _visualize_price_vs_rating(self, filtered_data, avg_price_by_rating):
        """Visualize price vs. rating relationship"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot
        ax1.scatter(filtered_data['rating'], filtered_data['price'], alpha=0.5, color='purple')
        ax1.set_title('Price vs. Rating Scatter Plot')
        ax1.set_xlabel('Rating')
        ax1.set_ylabel('Price (₹)')
        ax1.grid(True)

        # Price by rating box plot
        sns.boxplot(x='rating', y='price', data=filtered_data, ax=ax2)
        ax2.set_title('Price Distribution by Rating')
        ax2.set_xlabel('Rating')
        ax2.set_ylabel('Price (₹)')

        plt.tight_layout()
        plt.savefig('price_vs_rating.png')
        plt.close()

        # Line plot for average price by rating
        plt.figure(figsize=(10, 6))
        plt.plot(avg_price_by_rating['rating'], avg_price_by_rating['price'], marker='o', linewidth=2)
        plt.title('Average Price by Rating')
        plt.xlabel('Rating')
        plt.ylabel('Average Price (₹)')
        plt.grid(True)
        plt.savefig('avg_price_by_rating.png')
        plt.close()

    def analyze_review_rating_distribution(self):
        """Analyze distribution of reviews and ratings"""
        logger.info("Performing review and rating distribution analysis")

        # Check if required columns exist
        if 'reviews' not in self.data.columns or 'rating' not in self.data.columns:
            logger.error("Reviews or rating column not found in the dataset")
            return

        # Filter out missing values
        filtered_data = self.data[self.data['reviews'].notna() & self.data['rating'].notna()]

        # Get distribution of ratings
        rating_distribution = filtered_data['rating'].value_counts().sort_index()

        # Bin the review counts for distribution analysis
        filtered_data['review_bin'] = pd.cut(
            filtered_data['reviews'],
            bins=[0, 10, 50, 100, 500, 1000, 5000, 10000, float('inf')],
            labels=['0-10', '11-50', '51-100', '101-500', '501-1000', '1001-5000', '5001-10000', '10000+']
        )
        review_distribution = filtered_data['review_bin'].value_counts().sort_index()

        print("\n=== Review & Rating Distribution Analysis ===")
        print("Rating Distribution:")
        print(rating_distribution)

        print("\nReview Count Distribution:")
        print(review_distribution)

        # Calculate average rating
        avg_rating = filtered_data['rating'].mean()
        print(f"\nAverage Product Rating: {avg_rating:.2f}/5.0")

        # Calculate median number of reviews
        median_reviews = filtered_data['reviews'].median()
        print(f"Median Number of Reviews: {median_reviews:.0f}")

        # Analyze highly rated products
        highly_rated = filtered_data[filtered_data['rating'] >= 4.5]
        pct_highly_rated = (len(highly_rated) / len(filtered_data)) * 100
        print(f"Percentage of Highly Rated Products (4.5+ stars): {pct_highly_rated:.2f}%")

        # Visualize distributions
        self._visualize_review_rating_distribution(rating_distribution, review_distribution, filtered_data)

        print("\nAnalysis complete!")
        return rating_distribution, review_distribution

    def _visualize_review_rating_distribution(self, rating_distribution, review_distribution, filtered_data):
        """Visualize distribution of ratings and reviews"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Rating distribution
        rating_distribution.plot(kind='bar', ax=ax1, color='teal')
        ax1.set_title('Distribution of Product Ratings')
        ax1.set_xlabel('Rating')
        ax1.set_ylabel('Number of Products')
        ax1.grid(True, axis='y')

        # Review distribution
        review_distribution.plot(kind='bar', ax=ax2, color='coral')
        ax2.set_title('Distribution of Review Counts')
        ax2.set_xlabel('Number of Reviews')
        ax2.set_ylabel('Number of Products')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig('review_rating_distribution.png')
        plt.close()

        # Scatter plot - Rating vs Reviews with size representing price
        plt.figure(figsize=(10, 6))
        plt.scatter(
            filtered_data['rating'],
            filtered_data['reviews'],
            alpha=0.6,
            c=filtered_data['price'] if 'price' in filtered_data.columns else 'blue',
            cmap='viridis'
        )
        plt.colorbar(label='Price (₹)')
        plt.title('Rating vs. Number of Reviews')
        plt.xlabel('Rating')
        plt.ylabel('Number of Reviews')
        plt.grid(True)
        plt.yscale('log')  # Log scale for better visualization
        plt.savefig('rating_vs_reviews.png')
        plt.close()

    def analyze_sponsored_vs_organic(self):
        """Analyze differences between sponsored and organic products"""
        logger.info("Performing sponsored vs. organic analysis")

        # Check if required column exists
        if 'is_sponsored' not in self.data.columns:
            logger.error("Sponsored status column not found in the dataset")
            return

        # Count sponsored and organic products
        sponsored_count = self.data['is_sponsored'].sum()
        organic_count = len(self.data) - sponsored_count

        print("\n=== Sponsored vs. Organic Analysis ===")
        print(f"Sponsored Products: {sponsored_count}")
        print(f"Organic Products: {organic_count}")
        print(f"Percentage Sponsored: {(sponsored_count / len(self.data)) * 100:.2f}%")

        # Compare metrics
        comparison = {}

        # Convert boolean to categorical for groupby
        self.data['product_type'] = self.data['is_sponsored'].apply(lambda x: 'Sponsored' if x else 'Organic')

        # Compare average price
        if 'price' in self.data.columns:
            price_comparison = self.data.groupby('product_type')['price'].mean()
            comparison['price'] = price_comparison
            print("\nAverage Price:")
            print(price_comparison)

        # Compare average rating
        if 'rating' in self.data.columns:
            rating_comparison = self.data.groupby('product_type')['rating'].mean()
            comparison['rating'] = rating_comparison
            print("\nAverage Rating:")
            print(rating_comparison)

        # Compare average reviews
        if 'reviews' in self.data.columns:
            reviews_comparison = self.data.groupby('product_type')['reviews'].mean()
            comparison['reviews'] = reviews_comparison
            print("\nAverage Number of Reviews:")
            print(reviews_comparison)

        # Compare badge distribution
        if 'badges' in self.data.columns:
            badge_counts = self.data.groupby(['product_type', 'badges']).size().unstack(fill_value=0)
            if not badge_counts.empty:
                print("\nBadge Distribution:")
                print(badge_counts)
                comparison['badges'] = badge_counts

        # Visualize comparison
        self._visualize_sponsored_vs_organic(sponsored_count, organic_count, comparison)

        print("\nAnalysis complete!")
        return sponsored_count, organic_count, comparison

    def _visualize_sponsored_vs_organic(self, sponsored_count, organic_count, comparison):
        """Visualize sponsored vs. organic comparison"""
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Count distribution
        axes[0, 0].bar(['Sponsored', 'Organic'], [sponsored_count, organic_count], color=['orange', 'green'])
        axes[0, 0].set_title('Count of Sponsored vs. Organic Products')
        axes[0, 0].set_ylabel('Number of Products')
        axes[0, 0].grid(True, axis='y')

        # Plot 2: Price comparison if available
        if 'price' in comparison:
            comparison['price'].plot(kind='bar', ax=axes[0, 1], color=['orange', 'green'])
            axes[0, 1].set_title('Average Price: Sponsored vs. Organic')
            axes[0, 1].set_ylabel('Price (₹)')
            axes[0, 1].grid(True, axis='y')

        # Plot 3: Rating comparison if available
        if 'rating' in comparison:
            comparison['rating'].plot(kind='bar', ax=axes[1, 0], color=['orange', 'green'])
            axes[1, 0].set_title('Average Rating: Sponsored vs. Organic')
            axes[1, 0].set_ylabel('Average Rating')
            axes[1, 0].set_ylim(0, 5)
            axes[1, 0].grid(True, axis='y')

        # Plot 4: Reviews comparison if available
        if 'reviews' in comparison:
            comparison['reviews'].plot(kind='bar', ax=axes[1, 1], color=['orange', 'green'])
            axes[1, 1].set_title('Average Reviews: Sponsored vs. Organic')
            axes[1, 1].set_ylabel('Number of Reviews')
            axes[1, 1].grid(True, axis='y')

        plt.tight_layout()
        plt.savefig('sponsored_vs_organic.png')
        plt.close()

    def generate_full_report(self, output_dir="analysis_report"):
        """Generate a comprehensive analysis report"""
        logger.info("Generating full analysis report")

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Run all analyses
        brand_results = self.analyze_brand_performance()
        price_rating_results = self.analyze_price_vs_rating()
        review_rating_results = self.analyze_review_rating_distribution()
        sponsored_organic_results = self.analyze_sponsored_vs_organic()

        # Generate HTML report
        html_report = self._generate_html_report()

        # Save report
        with open(f"{output_dir}/report.html", "w", encoding="utf-8") as f:
            f.write(html_report)

        # Copy images to output directory
        for img in ['brand_performance.png', 'price_vs_rating.png', 'avg_price_by_rating.png',
                   'review_rating_distribution.png', 'rating_vs_reviews.png', 'sponsored_vs_organic.png']:
            if os.path.exists(img):
                import shutil
                shutil.copy(img, f"{output_dir}/{img}")

        # Generate summary statistics
        summary = self._generate_summary_stats()
        with open(f"{output_dir}/summary_stats.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        logger.info(f"Full report generated in '{output_dir}' directory")
        print(f"\nFull report generated in '{output_dir}' directory")
        return output_dir

    def _generate_summary_stats(self):
        """Generate summary statistics for the dataset"""
        summary = {
            "total_products": len(self.data),
            "date_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Price statistics
        if 'price' in self.data.columns:
            price_data = self.data['price'].dropna()
            summary["price"] = {
                "min": float(price_data.min()),
                "max": float(price_data.max()),
                "mean": float(price_data.mean()),
                "median": float(price_data.median())
            }

        # Rating statistics
        if 'rating' in self.data.columns:
            rating_data = self.data['rating'].dropna()
            summary["rating"] = {
                "mean": float(rating_data.mean()),
                "median": float(rating_data.median()),
                "most_common": float(rating_data.value_counts().idxmax())
            }

        # Reviews statistics
        if 'reviews' in self.data.columns:
            reviews_data = self.data['reviews'].dropna()
            summary["reviews"] = {
                "min": int(reviews_data.min()),
                "max": int(reviews_data.max()),
                "mean": float(reviews_data.mean()),
                "median": float(reviews_data.median())
            }

        # Top brands
        if 'brand' in self.data.columns:
            top_brands = self.data['brand'].value_counts().head(5).to_dict()
            summary["top_brands"] = top_brands

        # Sponsored percentage
        if 'is_sponsored' in self.data.columns:
            sponsored_pct = (self.data['is_sponsored'].sum() / len(self.data)) * 100
            summary["sponsored_percentage"] = float(sponsored_pct)

        return summary

    def _generate_html_report(self):
        """Generate HTML report content"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>amazon Product Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1 {{ color: #232F3E; }}
                h2 {{ color: #232F3E; margin-top: 30px; }}
                .section {{ margin-bottom: 40px; }}
                .image-container {{ margin: 20px 0; }}
                .image-container img {{ max-width: 100%; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .footer {{ margin-top: 40px; font-size: 0.8em; color: #666; }}
            </style>
        </head>
        <body>
            <h1>amazon Product Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Total Products Analyzed: {len(self.data)}</p>

            <div class="section">
                <h2>Brand Performance Analysis</h2>
                <div class="image-container">
                    <img src="brand_performance.png" alt="Brand Performance Analysis">
                </div>
            </div>

            <div class="section">
                <h2>Price vs. Rating Analysis</h2>
                <div class="image-container">
                    <img src="price_vs_rating.png" alt="Price vs Rating Scatter Plot">
                </div>
                <div class="image-container">
                    <img src="avg_price_by_rating.png" alt="Average Price by Rating">
                </div>
            </div>

            <div class="section">
                <h2>Review & Rating Distribution</h2>
                <div class="image-container">
                    <img src="review_rating_distribution.png" alt="Review and Rating Distribution">
                </div>
                <div class="image-container">
                    <img src="rating_vs_reviews.png" alt="Rating vs Number of Reviews">
                </div>
            </div>

            <div class="section">
                <h2>Sponsored vs. Organic Analysis</h2>
                <div class="image-container">
                    <img src="sponsored_vs_organic.png" alt="Sponsored vs Organic Comparison">
                </div>
            </div>

            <div class="footer">
                <p>This report was generated automatically by amazon Product Analyzer.</p>
            </div>
        </body>
        </html>
        """
        return html


def main():
    print("=== amazon Product Analyzer ===")
    print("1. Scrape new data")
    print("2. Load existing data")

    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        query = input("Enter product search query: ")
        pages = int(input("Enter number of pages to scrape (1-5): "))

        scraper = amazonScraper()
        products = scraper.search_products(query, max_pages=pages)

        if products:
            raw_file = scraper.save_to_csv("raw_amazon_products.csv")

            # Ask if user wants to scrape reviews for top products
            scrape_reviews = input("\nDo you want to scrape reviews for top products? (y/n): ")
            if scrape_reviews.lower() == 'y':
                # Get top 3 products by rating
                df = pd.DataFrame(products)
                if 'rating' in df.columns and 'asin' in df.columns:
                    top_products = df.sort_values('rating', ascending=False).head(3)

                    all_reviews = []
                    for idx, product in top_products.iterrows():
                        if pd.notna(product['asin']):
                            print(f"\nScraping reviews for: {product['title'][:50]}...")
                            reviews = scraper.scrape_product_reviews(product['asin'], max_pages=2)
                            all_reviews.extend(reviews)

                    if all_reviews:
                        reviews_df = pd.DataFrame(all_reviews)
                        reviews_file = "product_reviews.csv"
                        reviews_df.to_csv(reviews_file, index=False)
                        print(f"Saved {len(all_reviews)} reviews to {reviews_file}")

            input_data = raw_file
        else:
            print("No products found. Exiting...")
            return
    else:
        input_data = input("Enter path to existing CSV file: ")
        if not os.path.exists(input_data):
            print(f"File not found: {input_data}")
            return

    # Clean data
    print("\nCleaning data...")
    cleaner = DataCleaner(input_data)
    cleaned_data = cleaner.clean_data()
    cleaned_file = cleaner.save_to_csv()

    # Analyze data
    analyzer = DataAnalyzer(cleaned_data)

    print("\nWhat analysis would you like to perform?")
    print("1. Brand Performance Analysis")
    print("2. Price vs. Rating Analysis")
    print("3. Review & Rating Distribution Analysis")
    print("4. Sponsored vs. Organic Analysis")
    print("5. Full Report")

    analysis_choice = input("Enter your choice (1-5): ")

    if analysis_choice == '1':
        analyzer.analyze_brand_performance()
    elif analysis_choice == '2':
        analyzer.analyze_price_vs_rating()
    elif analysis_choice == '3':
        analyzer.analyze_review_rating_distribution()
    elif analysis_choice == '4':
        analyzer.analyze_sponsored_vs_organic()
    elif analysis_choice == '5':
        report_dir = input("Enter output directory name (default: analysis_report): ") or "analysis_report"
        analyzer.generate_full_report(output_dir=report_dir)
    else:
        print("Invalid choice. Exiting...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"\nAn error occurred: {e}")