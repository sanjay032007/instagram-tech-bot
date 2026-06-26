import os
import json
import time
import urllib.request
import urllib.parse
import feedparser
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

# --- Config & Secrets ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY")
IG_ACCESS_TOKEN = os.environ.get("IG_ACCESS_TOKEN")
IG_ACCOUNT_ID = os.environ.get("IG_ACCOUNT_ID")

def get_latest_news():
    print("Fetching top tech news from RSS...")
    # Fetching from TechCrunch or similar popular RSS
    feed = feedparser.parse('https://techcrunch.com/feed/')
    news_items = []
    for entry in feed.entries[:10]:
        news_items.append(f"Title: {entry.title}\nSummary: {entry.get('summary', '')}")
    return "\n\n".join(news_items)

def generate_post_content(news_text):
    print("Sending news to Gemini API to generate slides...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    system_prompt = """You are an expert Instagram tech news curator. 
Review the provided recent tech news. Pick the single most viral, breaking, or important story.
Create a 5-slide carousel post about it.
For each slide, provide a 'headline' (UPPERCASE) and 'subtext' (Sentence case).
CRITICAL STYLING: You MUST use ** tags to wrap exactly 1 or 2 words in each headline that should be colored with an accent color. (e.g., GLOBAL TECH\n**SELL-OFF.**)
Provide an 'unsplash_search_term' (e.g., 'microchip', 'data center', 'hacker') to find related background images.
Provide an Instagram 'caption' with relevant hashtags.
Output ONLY raw JSON using this schema:
{
  "unsplash_search_term": "string",
  "slides": [
    {"headline": "string", "subtext": "string"}
  ],
  "caption": "string"
}"""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=news_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.7
        )
    )
    
    return json.loads(response.text)

def download_unsplash_images(query, count=5):
    print(f"Fetching {count} images from Unsplash for query: {query}")
    url = f"https://api.unsplash.com/search/photos?query={urllib.parse.quote(query)}&per_page={count}&orientation=squarish&client_id={UNSPLASH_ACCESS_KEY}"
    
    req = urllib.request.Request(url)
    images = []
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results = data.get('results', [])
            for i, res in enumerate(results):
                img_url = res['urls']['regular']
                filename = f"raw_image_{i+1}.jpg"
                urllib.request.urlretrieve(img_url, filename)
                images.append(filename)
                print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Failed to fetch from Unsplash: {e}")
        # Fallback to a static generic tech image if Unsplash fails/limits
        fallback = "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=1000&auto=format&fit=crop"
        for i in range(count):
            filename = f"raw_image_{i+1}.jpg"
            urllib.request.urlretrieve(fallback, filename)
            images.append(filename)
    return images

def draw_styled_text(draw, text, font_bold, font_reg, default_color, accent_color, max_width, start_x, start_y, line_spacing=1.1):
    y = start_y
    lines = text.split('\n')
    bold_mode = False
    
    bbox_height_test = draw.textbbox((0, 0), "TEST", font=font_bold)
    line_height = bbox_height_test[3] - bbox_height_test[1]
    
    for line in lines:
        x = start_x
        words = line.split(' ')
        for word in words:
            if not word:
                continue
            has_start = word.startswith('**')
            has_end = False
            for ending in ['**', '**.', '**?', '**,', '**!', '**:']:
                if word.endswith(ending):
                    has_end = True
                    break
            
            clean_word = word.replace('**', '')
            if has_start:
                bold_mode = True
            is_accent = bold_mode
            current_font = font_bold if is_accent or clean_word.isupper() else font_reg
            current_color = accent_color if is_accent else default_color
            
            draw.text((x, y), clean_word, font=current_font, fill=current_color)
            w_bbox = draw.textbbox((0, 0), clean_word, font=current_font)
            word_width = w_bbox[2] - w_bbox[0]
            space_bbox = draw.textbbox((0, 0), " ", font=current_font)
            space_width = space_bbox[2] - space_bbox[0]
            
            x += word_width + space_width
            if has_end:
                bold_mode = False
        y += int(line_height * line_spacing)
    return y

def create_slides(content, image_paths):
    print("Generating slide images...")
    
    font_dir = "fonts/Inter Desktop/" if os.path.exists("fonts/Inter Desktop/") else ""
    try:
        font_bold = ImageFont.truetype(os.path.join(font_dir, "Inter-Bold.otf"), 68)
        font_reg = ImageFont.truetype(os.path.join(font_dir, "Inter-Regular.otf"), 68)
        font_sub = ImageFont.truetype(os.path.join(font_dir, "Inter-Regular.otf"), 34)
        font_brand = ImageFont.truetype(os.path.join(font_dir, "Inter-Bold.otf"), 22)
        font_num = ImageFont.truetype(os.path.join(font_dir, "Inter-Regular.otf"), 22)
    except:
        print("Using default fonts")
        font_bold = font_reg = font_sub = font_brand = font_num = ImageFont.load_default()

    slides_info = content['slides']
    final_slide_paths = []
    
    for idx, slide_info in enumerate(slides_info):
        width, height = 1080, 1080
        bg_path = image_paths[idx] if idx < len(image_paths) else image_paths[0]
        
        try:
            bg = Image.open(bg_path).convert("RGB")
        except:
            bg = Image.new("RGB", (width, height), (20, 20, 20))
            
        bg_w, bg_h = bg.size
        min_dim = min(bg_w, bg_h)
        crop_box = ((bg_w - min_dim)//2, (bg_h - min_dim)//2, (bg_w + min_dim)//2, (bg_h + min_dim)//2)
        bg = bg.crop(crop_box).resize((width, height), Image.Resampling.LANCZOS)
        
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        for y in range(height):
            if y < height // 2:
                opacity = int(230 - (y / (height // 2)) * 120)
            else:
                opacity = int(110 + ((y - height // 2) / (height // 2)) * 120)
            draw_overlay.line([(0, y), (width, y)], fill=(6, 6, 8, opacity))
            
        slide = Image.alpha_composite(bg.convert("RGBA"), overlay)
        draw = ImageDraw.Draw(slide)
        
        head_x, head_y = 90, 120
        max_text_width = width - 180
        accent_color = (0, 229, 255, 255) # Default cyan
        
        next_y = draw_styled_text(
            draw=draw, text=slide_info["headline"], font_bold=font_bold, font_reg=font_reg,
            default_color=(255, 255, 255, 255), accent_color=accent_color,
            max_width=max_text_width, start_x=head_x, start_y=head_y
        )
        
        sub_x, sub_y = 90, next_y + 35
        wrapped_lines = []
        current_line = []
        for word in slide_info["subtext"].split(' '):
            test_line = ' '.join(current_line + [word])
            if draw.textbbox((0, 0), test_line, font=font_sub)[2] <= max_text_width:
                current_line.append(word)
            else:
                wrapped_lines.append(' '.join(current_line))
                current_line = [word]
        if current_line: wrapped_lines.append(' '.join(current_line))
            
        sub_y_curr = sub_y
        for line in wrapped_lines:
            draw.text((sub_x, sub_y_curr), line, font=font_sub, fill=(230, 230, 235, 255))
            sub_y_curr += int((draw.textbbox((0, 0), line, font=font_sub)[3] - draw.textbbox((0,0), line, font=font_sub)[1]) * 1.3)
            
        draw.text((head_x, height - 70), "TECH NEWS TODAY", font=font_brand, fill=(255, 255, 255, 120))
        num_text = f"{idx+1:02d} / {len(slides_info):02d}"
        num_w = draw.textbbox((0,0), num_text, font=font_num)[2]
        draw.text((width - head_x - num_w, height - 70), num_text, font=font_num, fill=(255, 255, 255, 120))
        
        out_path = f"slide_{idx+1}.png"
        slide.convert("RGB").save(out_path)
        final_slide_paths.append(out_path)
        
    return final_slide_paths

# --- Instagram API Functions ---
def upload_to_catbox(file_path):
    print(f"Uploading {file_path} to catbox...")
    url = 'https://catbox.moe/user/api.php'
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    with open(file_path, 'rb') as f:
        file_content = f.read()
    filename = os.path.basename(file_path)
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="reqtype"\r\n\r\nfileupload\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="fileToUpload"; filename="{filename}"\r\n'
        f'Content-Type: image/png\r\n\r\n'
    ).encode('utf-8') + file_content + f'\r\n--{boundary}--\r\n'.encode('utf-8')
    
    req = urllib.request.Request(url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    req.add_header('User-Agent', 'Mozilla/5.0')
    try:
        with urllib.request.urlopen(req) as res:
            return res.read().decode('utf-8')
    except:
        return None

def post_to_instagram(image_urls, caption):
    print("Posting to Instagram...")
    item_ids = []
    # Create item containers
    for url in image_urls:
        req_url = f"https://graph.instagram.com/v20.0/{IG_ACCOUNT_ID}/media"
        data = urllib.parse.urlencode({'image_url': url, 'is_carousel_item': 'true', 'access_token': IG_ACCESS_TOKEN}).encode('utf-8')
        try:
            with urllib.request.urlopen(urllib.request.Request(req_url, data=data)) as res:
                item_ids.append(json.loads(res.read().decode())['id'])
        except Exception as e:
            print(f"Item error: {e}")
            return False
        time.sleep(2)
        
    # Create carousel
    req_url = f"https://graph.instagram.com/v20.0/{IG_ACCOUNT_ID}/media"
    data = urllib.parse.urlencode({'media_type': 'CAROUSEL', 'children': ','.join(item_ids), 'caption': caption, 'access_token': IG_ACCESS_TOKEN}).encode('utf-8')
    try:
        with urllib.request.urlopen(urllib.request.Request(req_url, data=data)) as res:
            carousel_id = json.loads(res.read().decode())['id']
    except Exception as e:
        print(f"Carousel error: {e}")
        return False
        
    # Wait for ready
    status_url = f"https://graph.instagram.com/v20.0/{carousel_id}?fields=status_code&access_token={IG_ACCESS_TOKEN}"
    while True:
        try:
            with urllib.request.urlopen(urllib.request.Request(status_url)) as res:
                if json.loads(res.read().decode())['status_code'] == 'FINISHED': break
        except: pass
        time.sleep(3)
        
    # Publish
    pub_url = f"https://graph.instagram.com/v20.0/{IG_ACCOUNT_ID}/media_publish"
    data = urllib.parse.urlencode({'creation_id': carousel_id, 'access_token': IG_ACCESS_TOKEN}).encode('utf-8')
    try:
        with urllib.request.urlopen(urllib.request.Request(pub_url, data=data)) as res:
            pub_id = json.loads(res.read().decode())['id']
            print(f"SUCCESS! Published post ID: {pub_id}")
            return True
    except Exception as e:
        print(f"Publish error: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    try:
        news_text = get_latest_news()
        content = generate_post_content(news_text)
        print("Generated Content:", json.dumps(content, indent=2))
        
        raw_images = download_unsplash_images(content['unsplash_search_term'])
        slide_paths = create_slides(content, raw_images)
        
        catbox_urls = []
        for path in slide_paths:
            catbox_urls.append(upload_to_catbox(path))
            
        if all(catbox_urls):
            post_to_instagram(catbox_urls, content['caption'])
        else:
            print("Failed to upload all images to Catbox.")
    except Exception as e:
        print(f"Workflow failed: {e}")
