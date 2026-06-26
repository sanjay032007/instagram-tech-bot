import os
import sys
import json
import time
import urllib.request
import urllib.parse
import feedparser
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from google import genai
from google.genai import types

# --- Config & Secrets ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY")
IG_ACCESS_TOKEN = os.environ.get("IG_ACCESS_TOKEN")
IG_ACCOUNT_ID = os.environ.get("IG_ACCOUNT_ID")
HISTORY_FILE = "used_images.txt"

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        for item in history[-50:]:
            f.write(f"{item}\n")

def get_latest_news():
    print("Fetching top tech news from RSS...")
    feed = feedparser.parse('https://techcrunch.com/feed/')
    news_items = []
    for entry in feed.entries[:10]:
        news_items.append(f"Title: {entry.title}\nSummary: {entry.get('summary', '')}")
    return "\n\n".join(news_items)

def generate_post_content(news_text):
    print("Sending news to Gemini API to extract entities and generate slides...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    system_prompt = """You are an expert Instagram tech news curator. 
Review the provided recent tech news. Pick the single most viral, breaking, or important story.
Extract the core 'news_topic' (e.g. 'OpenAI delays GPT-5.6 due to safety concerns').
Create an array of 5 'search_queries' for Unsplash, ordered from most specific to least specific. Use extracted entities (e.g., 'OpenAI headquarters', 'Sam Altman', 'AI safety'). Do NOT use generic terms like 'technology' unless absolutely necessary.
Create a 5-slide carousel post about it. For each slide, provide a 'headline' (UPPERCASE) and 'subtext' (Sentence case).
CRITICAL STYLING: You MUST use ** tags to wrap exactly 1 or 2 words in each headline that should be colored with an accent color. (e.g., GLOBAL TECH\n**SELL-OFF.**)
Provide an Instagram 'caption' with relevant hashtags.
Output ONLY raw JSON using this schema:
{
  "news_topic": "string",
  "search_queries": ["string", "string", "string", "string", "string"],
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

def validate_image_with_gemini(image_path, news_topic):
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"Validating image relevance to: '{news_topic}'")
    try:
        myfile = client.files.upload(file=image_path)
        prompt = f"""You are a professional editorial image reviewer.
Analyze this image. Does it look like a high-quality, professional photograph or highly relevant illustration for a news story about: '{news_topic}'?
Reject images that are: abstract particle art, random cubes, random gradients, unrelated stock photos, or extremely generic.
Score the relevance and quality from 0 to 100.
Output ONLY raw JSON format: {{"score": 85, "reason": "Clear photo, highly relevant."}}"""

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[myfile, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        validation = json.loads(response.text)
        print(f"Validation Score: {validation.get('score')} | Reason: {validation.get('reason')}")
        return validation.get('score', 0)
    except Exception as e:
        print(f"Validation error: {e}")
        return 0

def get_valid_unsplash_image(search_queries, news_topic):
    history = load_history()
    for query in search_queries:
        print(f"\n--- SEARCH STAGE: '{query}' ---")
        url = f"https://api.unsplash.com/search/photos?query={urllib.parse.quote(query)}&per_page=5&orientation=landscape&client_id={UNSPLASH_ACCESS_KEY}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                results = json.loads(response.read().decode()).get('results', [])
                for res in results:
                    img_id = res['id']
                    if img_id in history:
                        print(f"Skipping {img_id}: Found in recent history.")
                        continue
                    
                    img_url = res['urls']['regular']
                    print(f"Evaluating: {img_url}")
                    temp_path = f"temp_{img_id}.jpg"
                    urllib.request.urlretrieve(img_url, temp_path)
                    
                    score = validate_image_with_gemini(temp_path, news_topic)
                    if score >= 80:
                        print(f"ACCEPTED Image {img_id} (Score: {score})")
                        history.append(img_id)
                        save_history(history)
                        return temp_path
                    else:
                        print(f"REJECTED Image {img_id} (Score: {score})")
                        os.remove(temp_path)
        except Exception as e:
            print(f"Search error for query '{query}': {e}")
    print("\nFAILED: All search queries exhausted. No valid images found on Unsplash.")
    return None

def generate_fallback_image(news_topic):
    print(f"\n--- FALLBACK STAGE: Generating AI Image for '{news_topic}' ---")
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        result = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=f"A photorealistic, clean, editorial illustration about: {news_topic}. Professional, high quality.",
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/jpeg",
                aspect_ratio="1:1"
            )
        )
        for generated_image in result.generated_images:
            image = Image.open(io.BytesIO(generated_image.image.image_bytes))
            path = 'fallback_image.jpg'
            image.save(path)
            print("ACCEPTED AI Fallback Image.")
            # Record a unique ID so we track it (just a timestamp)
            history = load_history()
            history.append(f"ai_gen_{int(time.time())}")
            save_history(history)
            return path
    except Exception as e:
        print(f"Fallback Imagen error: {e}")
    return None

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

def create_slides(content, bg_path):
    print("\nGenerating slide images...")
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
    width, height = 1080, 1080
    
    try:
        base_bg = Image.open(bg_path).convert("RGB")
        bg_w, bg_h = base_bg.size
        min_dim = min(bg_w, bg_h)
        crop_box = ((bg_w - min_dim)//2, (bg_h - min_dim)//2, (bg_w + min_dim)//2, (bg_h + min_dim)//2)
        base_bg = base_bg.crop(crop_box).resize((width, height), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error loading background image: {e}")
        base_bg = Image.new("RGB", (width, height), (20, 20, 20))
        
    for idx, slide_info in enumerate(slides_info):
        # We use the identical base_bg for all slides for consistency
        bg = base_bg.copy()
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

def upload_image(file_path):
    print(f"Uploading {file_path} to freeimage.host...")
    url = "https://freeimage.host/api/1/upload"
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data = urllib.parse.urlencode({
        "key": "6d207e02198a847aa98d0a2a901485a5",
        "action": "upload",
        "source": b64,
        "format": "json"
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req) as res:
            response = json.loads(res.read().decode())
            print(f"Upload success: {response['image']['url']}")
            return response["image"]["url"]
    except Exception as e:
        print(f"Upload failed: {e}")
        return None

def post_to_instagram(image_urls, caption):
    print("Posting to Instagram...")
    item_ids = []
    for url in image_urls:
        req_url = f"https://graph.instagram.com/v20.0/{IG_ACCOUNT_ID}/media"
        data = urllib.parse.urlencode({'image_url': url, 'is_carousel_item': 'true', 'access_token': IG_ACCESS_TOKEN}).encode('utf-8')
        try:
            with urllib.request.urlopen(urllib.request.Request(req_url, data=data)) as res:
                item_ids.append(json.loads(res.read().decode())['id'])
        except urllib.error.HTTPError as e:
            print(f"IG Item HTTPError {e.code}: {e.read().decode('utf-8')}")
            return False
        except Exception as e:
            print(f"Item error: {e}")
            return False
        time.sleep(2)
        
    req_url = f"https://graph.instagram.com/v20.0/{IG_ACCOUNT_ID}/media"
    data = urllib.parse.urlencode({'media_type': 'CAROUSEL', 'children': ','.join(item_ids), 'caption': caption, 'access_token': IG_ACCESS_TOKEN}).encode('utf-8')
    try:
        with urllib.request.urlopen(urllib.request.Request(req_url, data=data)) as res:
            carousel_id = json.loads(res.read().decode())['id']
    except urllib.error.HTTPError as e:
        print(f"IG Carousel HTTPError {e.code}: {e.read().decode('utf-8')}")
        return False
    except Exception as e:
        print(f"Carousel error: {e}")
        return False
        
    status_url = f"https://graph.instagram.com/v20.0/{carousel_id}?fields=status_code&access_token={IG_ACCESS_TOKEN}"
    while True:
        try:
            with urllib.request.urlopen(urllib.request.Request(status_url)) as res:
                if json.loads(res.read().decode())['status_code'] == 'FINISHED': break
        except: pass
        time.sleep(3)
        
    pub_url = f"https://graph.instagram.com/v20.0/{IG_ACCOUNT_ID}/media_publish"
    data = urllib.parse.urlencode({'creation_id': carousel_id, 'access_token': IG_ACCESS_TOKEN}).encode('utf-8')
    try:
        with urllib.request.urlopen(urllib.request.Request(pub_url, data=data)) as res:
            pub_id = json.loads(res.read().decode())['id']
            print(f"SUCCESS! Published post ID: {pub_id}")
            return True
    except urllib.error.HTTPError as e:
        print(f"IG Publish HTTPError {e.code}: {e.read().decode('utf-8')}")
        return False
    except Exception as e:
        print(f"Publish error: {e}")
        return False

if __name__ == "__main__":
    try:
        news_text = get_latest_news()
        content = generate_post_content(news_text)
        print("Generated Content:", json.dumps(content, indent=2))
        
        news_topic = content.get('news_topic', 'Technology news')
        queries = content.get('search_queries', ['technology'])
        
        selected_image_path = get_valid_unsplash_image(queries, news_topic)
        if not selected_image_path:
            selected_image_path = generate_fallback_image(news_topic)
            
        if not selected_image_path:
            print("FATAL ERROR: Failed to find or generate any valid images for this post. Aborting to maintain quality.")
            sys.exit(1)
            
        slide_paths = create_slides(content, selected_image_path)
        
        image_urls = []
        for path in slide_paths:
            image_urls.append(upload_image(path))
            
        if all(image_urls):
            success = post_to_instagram(image_urls, content['caption'])
            if not success:
                sys.exit(1)
        else:
            print("Failed to upload all slides.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Workflow failed: {e}")
        sys.exit(1)
