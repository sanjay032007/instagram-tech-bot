import os
import sys
import json
import time
import urllib.request
import urllib.parse
import feedparser
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
import base64
from google import genai
from google.genai import types
from tenacity import retry, wait_fixed, stop_after_attempt

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

@retry(wait=wait_fixed(25), stop=stop_after_attempt(4))
def run_with_failover(task_name, models, execution_func):
    """
    Tries a list of models sequentially. If a model fails (e.g., 429 Rate Limit),
    it immediately falls over to the next model. If all models fail, it raises
    an exception so Tenacity can trigger exponential backoff and retry the whole chain.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    last_exception = None
    for model in models:
        try:
            result = execution_func(client, model)
            print(f"[GEMINI] Task: {task_name} | Model Used: {model}")
            return result
        except Exception as e:
            err_str = str(e)
            if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                print(f"[GEMINI] Rate limited on {model}. Failing over to next model...")
            else:
                print(f"[GEMINI] Error on {model} ({e}). Failing over to next model...")
            last_exception = e
            continue
    
    print(f"[GEMINI] All configured models exhausted for task '{task_name}'. Triggering backoff retry...")
    raise last_exception

def generate_post_content(news_text):
    print("Sending news to Gemini API to extract entities and generate slides...")
    
    def exec_func(client, model):
        system_prompt = """You are an expert Instagram tech news curator. 
Review the provided recent tech news. Pick the single most viral, breaking, or important story.
Create a 5-slide carousel post about it.

CRITICAL TEXT LIMITS:
- Headline (UPPERCASE): Maximum 2 lines (approx 6-8 words).
- Subtext (Sentence case): Maximum 3-4 short lines (approx 20-30 words).
- You MUST use ** tags to wrap exactly 1 or 2 words in each headline that should be colored with an accent color. (e.g., GLOBAL TECH\n**SELL-OFF.**)

For EACH slide, also generate 3 highly specific 'search_queries' for Unsplash that relate specifically to that slide's content. Do NOT use generic terms like 'technology' unless absolutely necessary.

Provide an Instagram 'caption' with relevant hashtags.

Output ONLY raw JSON using this schema:
{
  "news_topic": "string",
  "slides": [
    {
      "headline": "string",
      "subtext": "string",
      "search_queries": ["query1", "query2", "query3"]
    }
  ],
  "caption": "string"
}"""
        response = client.models.generate_content(
            model=model,
            contents=news_text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.7
            )
        )
        return json.loads(response.text)

    models = ['gemini-3.0-flash-live', 'gemini-2.5-flash', 'gemini-2.5-flash-lite']
    try:
        return run_with_failover("News Generation & Keyword Extraction", models, exec_func)
    except Exception as e:
        print(f"Failed to generate content after complete failover and retries: {e}")
        raise e

def validate_image_with_gemini(image_path, slide_context):
    print(f"Validating image relevance to: '{slide_context}'")
    
    def exec_func(client, model):
        myfile = client.files.upload(file=image_path)
        prompt = f"""You are a professional editorial image reviewer.
Analyze this image. Does it look like a high-quality, professional photograph or highly relevant illustration for a slide discussing: '{slide_context}'?
Reject images that are: abstract particle art, random cubes, random gradients, unrelated stock photos, or extremely generic.
Score the relevance and quality from 0 to 100.
Output ONLY raw JSON format: {{"score": 85, "reason": "Clear photo, highly relevant."}}"""

        response = client.models.generate_content(
            model=model,
            contents=[myfile, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        validation = json.loads(response.text)
        print(f"Validation Details: {validation.get('score')} | {validation.get('reason')}")
        return validation.get('score', 0)

    models = ['gemini-3.0-flash-live', 'gemini-2.5-flash-lite', 'gemini-2.5-flash']
    try:
        return run_with_failover("Image Relevance Validation", models, exec_func)
    except Exception as e:
        print(f"Validation error (retries exhausted): {e}")
        return 0

def get_safe_zone(image_path):
    print(f"Calculating dynamic layout safe zone...")
    
    def exec_func(client, model):
        myfile = client.files.upload(file=image_path)
        prompt = """You are a layout designer. We need to place a text block on this 1080x1080 image.
Find the largest empty or 'safe' area that avoids covering human faces, important logos, or the main subject of the image.
The text block needs to occupy roughly 35-45% of the image.
Provide the X and Y coordinates of the top-left corner of this safe zone, and its maximum Width and Height.
Typically safe zones are at the top (y=100) or bottom (y=500), spanning the full width (x=80, w=920).
Output ONLY raw JSON format: {"x": 80, "y": 600, "w": 920, "h": 400}"""

        response = client.models.generate_content(
            model=model,
            contents=[myfile, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)

    models = ['gemini-3.0-flash-live', 'gemini-2.5-flash-lite', 'gemini-2.5-flash']
    try:
        data = run_with_failover("Layout Reasoning", models, exec_func)
        print(f"Calculated Safe Zone: {data}")
        return data
    except Exception as e:
        print(f"Safe zone calculation error (retries exhausted): {e}")
        return {"x": 80, "y": 550, "w": 920, "h": 450}

def rewrite_text(text, target_words):
    print(f"Rewriting text to fit within {target_words} words...")
    
    def exec_func(client, model):
        prompt = f"Rewrite this text to be shorter and punchier, fitting exactly within {target_words} words maximum while retaining core meaning: {text}"
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text.strip()

    models = ['gemini-3.0-flash-live', 'gemini-2.5-flash-lite', 'gemini-2.5-flash']
    try:
        new_text = run_with_failover("Typography Rewriting", models, exec_func)
        print(f"Rewrote to: {new_text}")
        return new_text
    except Exception as e:
        print(f"Rewrite error (retries exhausted): {e}")
        return text 

def get_valid_unsplash_image(search_queries, slide_context):
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
                    print(f"Evaluating candidate image: {img_url}")
                    temp_path = f"temp_{img_id}.jpg"
                    urllib.request.urlretrieve(img_url, temp_path)
                    
                    score = validate_image_with_gemini(temp_path, slide_context)
                    if score >= 80:
                        print(f"ACCEPTED Image {img_id} (Score: {score}). Breaking search loop immediately.")
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

@retry(wait=wait_fixed(25), stop=stop_after_attempt(4))
def generate_fallback_image(slide_context):
    print(f"\n--- FALLBACK STAGE: Generating AI Image for '{slide_context}' ---")
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        print(f"[GEMINI] Task: AI Image Generation | Model Used: imagen-3.0-generate-001")
        result = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=f"A photorealistic, clean, editorial illustration about: {slide_context}. Professional, high quality.",
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/jpeg",
                aspect_ratio="1:1"
            )
        )
        for generated_image in result.generated_images:
            image = Image.open(io.BytesIO(generated_image.image.image_bytes))
            path = f'fallback_image_{int(time.time())}.jpg'
            image.save(path)
            print("ACCEPTED AI Fallback Image.")
            history = load_history()
            history.append(path.split('.')[0])
            save_history(history)
            return path
    except Exception as e:
        print(f"Fallback Imagen error: {e}")
        raise e
    return None

def wrap_text_to_lines(draw, text, font, max_width):
    lines = []
    current_line = []
    for word in text.split(' '):
        test_line = ' '.join(current_line + [word])
        clean_line = test_line.replace('**', '')
        if draw.textbbox((0, 0), clean_line, font=font)[2] <= max_width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    return lines

def draw_styled_text_lines(draw, lines, font_bold, font_reg, default_color, accent_color, start_x, start_y, line_spacing=1.4):
    y = start_y
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

def draw_rounded_rectangle(draw, bounds, radius, fill):
    x0, y0, x1, y1 = bounds
    draw.rectangle([x0 + radius, y0, x1 - radius, y1], fill=fill)
    draw.rectangle([x0, y0 + radius, x1, y1 - radius], fill=fill)
    draw.pieslice([x0, y0, x0 + radius * 2, y0 + radius * 2], 180, 270, fill=fill)
    draw.pieslice([x1 - radius * 2, y0, x1, y0 + radius * 2], 270, 360, fill=fill)
    draw.pieslice([x0, y1 - radius * 2, x0 + radius * 2, y1], 90, 180, fill=fill)
    draw.pieslice([x1 - radius * 2, y1 - radius * 2, x1, y1], 0, 90, fill=fill)

def create_slides(content, slide_image_paths):
    print("\nGenerating slide images with Dynamic Layout & Advanced Typography...")
    font_dir = "fonts/Inter Desktop/" if os.path.exists("fonts/Inter Desktop/") else ""
    try:
        font_bold = ImageFont.truetype(os.path.join(font_dir, "Inter-Bold.otf"), 100) # MASSIVE
        font_reg = ImageFont.truetype(os.path.join(font_dir, "Inter-Regular.otf"), 100)
        font_sub = ImageFont.truetype(os.path.join(font_dir, "Inter-Regular.otf"), 46) # LARGE
        font_brand = ImageFont.truetype(os.path.join(font_dir, "Inter-Bold.otf"), 24)
        font_num = ImageFont.truetype(os.path.join(font_dir, "Inter-Regular.otf"), 24)
    except:
        print("Using default fonts")
        font_bold = font_reg = font_sub = font_brand = font_num = ImageFont.load_default()

    slides_info = content['slides']
    final_slide_paths = []
    width, height = 1080, 1080
    
    for idx, slide_info in enumerate(slides_info):
        bg_path = slide_image_paths[idx]
        try:
            base_bg = Image.open(bg_path).convert("RGB")
            bg_w, bg_h = base_bg.size
            min_dim = min(bg_w, bg_h)
            crop_box = ((bg_w - min_dim)//2, (bg_h - min_dim)//2, (bg_w + min_dim)//2, (bg_h + min_dim)//2)
            bg = base_bg.crop(crop_box).resize((width, height), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading background image {bg_path}: {e}")
            bg = Image.new("RGB", (width, height), (20, 20, 20))
            
        bg.save(f"temp_layout_{idx}.jpg")
        safe_zone = get_safe_zone(f"temp_layout_{idx}.jpg")
        
        sz_x = max(40, safe_zone.get("x", 80))
        sz_y = max(40, safe_zone.get("y", 550))
        sz_w = min(width - sz_x - 40, safe_zone.get("w", 920))
        sz_h = min(height - sz_y - 40, safe_zone.get("h", 450))
        
        dummy_img = Image.new("RGBA", (width, height))
        draw_measure = ImageDraw.Draw(dummy_img)
        
        headline_text = slide_info["headline"]
        subtext = slide_info["subtext"]
        
        max_headline_lines = 2
        max_subtext_lines = 3
        
        while True:
            head_lines = wrap_text_to_lines(draw_measure, headline_text, font_bold, sz_w - 80) 
            if len(head_lines) <= max_headline_lines:
                break
            headline_text = rewrite_text(headline_text, 6) 
            
        while True:
            sub_lines = wrap_text_to_lines(draw_measure, subtext, font_sub, sz_w - 80)
            if len(sub_lines) <= max_subtext_lines:
                break
            subtext = rewrite_text(subtext, 20) 

        h_line_height = (draw_measure.textbbox((0,0), "T", font=font_bold)[3] - draw_measure.textbbox((0,0), "T", font=font_bold)[1]) * 1.4
        s_line_height = (draw_measure.textbbox((0,0), "T", font=font_sub)[3] - draw_measure.textbbox((0,0), "T", font=font_sub)[1]) * 1.4
        
        total_text_height = (len(head_lines) * h_line_height) + 60 + (len(sub_lines) * s_line_height) + 60 
        
        final_y = sz_y + max(0, (sz_h - total_text_height) // 2)
        
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        bg_rect_bounds = [sz_x, final_y, sz_x + sz_w, final_y + total_text_height + 40]
        draw_rounded_rectangle(draw_overlay, bg_rect_bounds, radius=30, fill=(15, 15, 20, 210))
        
        draw_overlay.text((sz_x + 20, final_y - 40), "TECH NEWS TODAY", font=font_brand, fill=(255, 255, 255, 200))
        num_text = f"{idx+1:02d} / {len(slides_info):02d}"
        num_w = draw_overlay.textbbox((0,0), num_text, font=font_num)[2]
        draw_overlay.text((sz_x + sz_w - num_w - 20, final_y - 40), num_text, font=font_num, fill=(255, 255, 255, 200))

        slide = Image.alpha_composite(bg.convert("RGBA"), overlay)
        draw = ImageDraw.Draw(slide)
        
        start_x = sz_x + 40
        current_y = final_y + 40
        accent_color = (0, 229, 255, 255)
        
        current_y = draw_styled_text_lines(draw, head_lines, font_bold, font_reg, (255, 255, 255, 255), accent_color, start_x, current_y, 1.4)
        current_y += 20 
        draw_styled_text_lines(draw, sub_lines, font_sub, font_sub, (240, 240, 245, 255), accent_color, start_x, current_y, 1.4)
        
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
        
        slide_image_paths = []
        for idx, slide in enumerate(content['slides']):
            print(f"\n=======================")
            print(f"PROCESSING SLIDE {idx+1}")
            print(f"=======================")
            
            slide_context = f"{slide['headline']} - {slide['subtext']}"
            queries = slide.get('search_queries', [])
            
            img_path = get_valid_unsplash_image(queries, slide_context)
            if not img_path:
                img_path = generate_fallback_image(slide_context)
                
            if not img_path:
                print(f"FATAL ERROR: Failed to find or generate valid image for Slide {idx+1}. Aborting to maintain quality.")
                sys.exit(1)
            
            slide_image_paths.append(img_path)
            
        slide_paths = create_slides(content, slide_image_paths)
        
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
