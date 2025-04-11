import logging
import requests

from urllib.parse import urlparse, parse_qs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = 'AIzaSyDP4QKTPFmdTVBSaAgqFYBlm3uHbveGYcM'

def extract_video_id(url: str) -> str | None:
    if not url:
        return None
    # Examples:
    # https://www.youtube.com/watch?v=VIDEO_ID
    # https://youtu.be/VIDEO_ID
    # https://www.youtube.com/embed/VIDEO_ID
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        if parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        if parsed_url.path.startswith('/shorts/'):
             return parsed_url.path.split('/')[2]
    logger.warning(f"Could not extract video ID from URL: {url}")
    return None

def get_video_details(api_key: str, video_id: str) -> dict | None:
    if not api_key:
        logger.error("API key is missing.")
        return None
    if not video_id:
        logger.error("Video ID is missing.")
        return None

    base_url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        'key': api_key,
        'id': video_id,
        'part': 'snippet'
    }
    try:
        logger.info(f"Requesting video details for ID: {video_id}")
        response = requests.get(base_url, params=params)
        response.raise_for_status() 
        data = response.json()

        if 'items' in data and len(data['items']) > 0:
            snippet = data['items'][0].get('snippet')
            if snippet:
                thumbnail_url = snippet.get('thumbnails', {}).get('high', {}).get('url')
                if not thumbnail_url:
                     thumbnail_url = snippet.get('thumbnails', {}).get('medium', {}).get('url')
                if not thumbnail_url:
                     thumbnail_url = snippet.get('thumbnails', {}).get('default', {}).get('url')

                details = {
                    'title': snippet.get('title'),
                    'channel_title': snippet.get('channelTitle'),
                    'category_id': snippet.get('categoryId'),
                    'tags': snippet.get('tags', []), # Tags might be missing
                    'thumbnail_url': thumbnail_url
                }
                logger.info("Successfully retrieved video details.")
                return details
            else:
                 logger.warning(f"No snippet found in API response for video ID: {video_id}")
                 return None
        else:
            logger.warning(f"No items found in API response for video ID: {video_id}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        try:
            logger.error(f"Response Body: {response.text}")
        except NameError:
             pass
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None

def get_category_name(api_key: str, category_id: str, region_code: str = 'US') -> str | None:
    if not api_key:
        logger.error("API key is missing for category lookup.")
        return None
    if not category_id:
        logger.warning("Category ID is missing, cannot look up name.")
        return "Unknown"

    base_url = "https://www.googleapis.com/youtube/v3/videoCategories"
    params = {
        'key': api_key,
        'id': category_id,
        'part': 'snippet',
        'regionCode': region_code
    }
    try:
        logger.info(f"Requesting category name for ID: {category_id}")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if 'items' in data and len(data['items']) > 0:
            category_name = data['items'][0].get('snippet', {}).get('title')
            logger.info(f"Found category name: {category_name}")
            return category_name
        else:
             logger.warning(f"Category ID {category_id} not found in API response.")
             return "Unknown"

    except requests.exceptions.RequestException as e:
        logger.error(f"API request for category failed: {e}")
        return "Unknown"
    except Exception as e:
         logger.error(f"An unexpected error occurred getting category name: {e}")
         return "Unknown"

def get_prompt_for_model(youtube_url: str) -> str | None:
    if not youtube_url:
        logger.error("url is missing")
        return None
    video_id = extract_video_id(youtube_url)

    if video_id:
        details = get_video_details(API_KEY, video_id)

        if details:
            category_name = get_category_name(API_KEY, details.get('category_id'))
            bos_token_str = "<|endoftext|>"
            separator_str = "\nComment:\n"
        
            prompt_text = (
                f"{bos_token_str}\n"
                f"Title: {details.get('title', 'None')}\n"
                f"Channel: {details.get('channel_title', 'None')}\n"
                f"Category: {category_name if category_name else 'Unknown'}\n"
                f"Tags: {', '.join(details.get('tags', ['None']))}"
                f"{separator_str}"
            )
            return prompt_text
        else:
            logger.error(f"\nCould not retrieve details for video ID: {video_id}")
    else:
        logger.error(f"\nCould not extract Video ID from URL: {youtube_url}")
