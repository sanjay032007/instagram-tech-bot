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
GEMINI_API_KEY_2 = os.environ.get("GEMINI_API_KEY_2")
API_KEYS = [k for k in [GEMINI_API_KEY, GEMINI_API_KEY_2] if k]
UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY")
IG_ACCESS_TOKEN = os.environ.get("IG_ACCESS_TOKEN")
IG_ACCOUNT_ID = os.environ.get("IG_ACCOUNT_ID")
HISTORY_FILE = "used_images.txt"
NEWS_HISTORY_FILE = "used_news.txt"

# Colors
COLOR_WHITE = (255, 255, 255, 255)
COLOR_GOLD = (235, 203, 107, 255) # Premium Nvidia/Apple style gold

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        for item in history[-50:]:
            f.write(f"{item}\n")

def load_news_history():
    if not os.path.exists(NEWS_HISTORY_FILE):
        return []
    with open(NEWS_HISTORY_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def save_news_history(history):
    with open(NEWS_HISTORY_FILE, "w", encoding="utf-8") as f:
        for item in history[-50:]:
            f.write(f"{item}\n")

def get_latest_news():
    print("Fetching top tech news from RSS...")
    news_history = load_news_history()
    feed = feedparser.parse('https://techcrunch.com/feed/')
    news_items = []
    
    for entry in feed.entries:
        if entry.title.strip() not in news_history:
            news_items.append(f"Title: {entry.title}\nSummary: {entry.get('summary', '')}")
            if len(news_items) >= 10:
                break
                
    if not news_items:
        print("No new unposted news found! Exiting to save quota.")
        sys.exit(0)
        
    return "\n\n".join(news_items)

@retry(wait=wait_fixed(25), stop=stop_after_attempt(4))
def run_with_failover(task_name, models, execution_func):
    if not API_KEYS:
        raise ValueError("No API keys found for Gemini.")
        
    last_exception = None
    for api_key in API_KEYS:
        client = genai.Client(api_key=api_key)
        for model in models:
            try:
                result = execution_func(client, model)
                print(f"[GEMINI] Task: {task_name} | Model Used: {model}")
                return result
            except Exception as e:
                err_str = str(e).lower()
                if '429' in err_str or 'resource_exhausted' in err_str or 'quota' in err_str:
                    print(f"[GEMINI] Rate limited/Quota on {model}. Failing over to next model...")
                else:
                    print(f"[GEMINI] Error on {model} ({e}). Failing over to next model...")
                last_exception = e
                continue
        print(f"[GEMINI] All models exhausted on current API key. Switching to next key (if any)...")
            
    print(f"[GEMINI] All configured models and API keys exhausted for task '{task_name}'.")
    err_str = str(last_exception).lower()
    if '429' in err_str or 'quota' in err_str or 'resource_exhausted' in err_str:
        print("CRITICAL: Daily Quota completely exhausted across all models AND keys. Aborting to prevent wasted retries.")
        sys.exit(1)
        
    print("Triggering backoff retry for non-quota error...")
    raise last_exception

def generate_post_content(news_text):
    print("Sending news to Gemini API to extract entities and generate slides...")
    
    def exec_func(client, model):
        system_prompt = """You are an expert Instagram tech news curator.
Pick the single most viral, breaking, or important story from the provided news.

Create an Instagram carousel with 3 to 7 slides maximum (NEVER more than 7).
Decide the number of slides based on how complex and content-rich the topic is:
- Simple news (1 clear fact): 3-4 slides
- Medium topic (a few angles): 4-5 slides
- Complex/rich topic (many details, stats, comparisons): 6-7 slides

SLIDE STRUCTURE RULES:
- Slide 1: ALWAYS type "cover" — Large impactful headline, short subtext.
- Last slide: ALWAYS type "cta" — Call-to-action headline, subtext asks a question for comments.
- Middle slides: Choose the BEST type for each slide from the options below based on the topic.

SLIDE TYPES AND THEIR RULES:
1. "cover"    — Slide 1 only. Headline: max 3 lines, bold. Subtext: max 1 short line.
2. "context"  — A short question headline (e.g., "WHAT IS IT?", "WHY DOES IT MATTER?"). Subtext: 1 clear sentence.
3. "bullets"  — Headline is short. Provide EXACTLY 3 bullet_points. Each bullet: 3-7 words max.
4. "stat"     — A big impressive number/stat as headline (e.g., "$40B", "10x FASTER"). Subtext: what the stat means.
5. "quote"    — A powerful direct quote or key statement as headline. Subtext: who said it or context.
6. "comparison" — Headline like "BEFORE VS AFTER" or "OLD VS NEW". bullet_points: provide EXACTLY 2, one per side.
7. "cta"      — Last slide only. Headline: short call-to-action (e.g., "FOLLOW FOR MORE"). Subtext: engaging question.

CRITICAL FORMATTING RULES:
- Use ** tags around words in the headline that should be highlighted in GOLD color.
  Example: THE FUTURE OF\n**ARTIFICIAL**\n**INTELLIGENCE**
- Keep all text concise. No long paragraphs.
- For EACH slide, generate 3 highly specific Unsplash search queries deeply related to that slide's topic.
  Prioritize dark, cinematic, moody technical photography. Avoid generic terms like "technology" or "business".

Output ONLY raw JSON using this exact schema:
{
  "original_title": "exact title of the article you selected",
  "news_topic": "string",
  "slides": [
    {
      "slide_type": "cover|context|bullets|stat|quote|comparison|cta",
      "headline": "string",
      "subtext": "string",
      "bullet_points": ["point 1", "point 2", "point 3"],
      "search_queries": ["query1", "query2", "query3"]
    }
  ],
  "caption": "string"
}
Note: bullet_points is only required for slide_type 'bullets' and 'comparison'. Omit it for all other types."""
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

    models = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.0-pro-exp-02-05', 'gemini-2.0-flash', 'gemini-1.5-pro']
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
Analyze this image. Does it look like a high-quality, professional photograph or highly relevant illustration for a presentation slide discussing: '{slide_context}'?
We strongly prefer images with dark backgrounds or negative space at the top.
Score relevance and quality from 0 to 100.
Output ONLY raw JSON format: {{"score": 85, "reason": "Clear photo, dark background."}}"""
        response = client.models.generate_content(
            model=model,
            contents=[myfile, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        validation = json.loads(response.text)
        print(f"Validation Details: {validation.get('score')} | {validation.get('reason')}")
        return validation.get('score', 0)

    models = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.0-pro-exp-02-05', 'gemini-2.0-flash', 'gemini-1.5-pro']
    try:
        return run_with_failover("Image Relevance Validation", models, exec_func)
    except:
        return 0



def get_valid_unsplash_image(search_queries, slide_context):
    if not UNSPLASH_ACCESS_KEY:
        print("No Unsplash key found. Skipping search.")
        return None
    history = load_history()
    evaluations = 0
    
    for query in search_queries:
        print(f"\n--- SEARCH STAGE: '{query}' ---")
        url = f"https://api.unsplash.com/search/photos?query={urllib.parse.quote(query)}&per_page=5&orientation=landscape&client_id={UNSPLASH_ACCESS_KEY}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                results = json.loads(response.read().decode()).get('results', [])
                for res in results:
                    if evaluations >= 3:
                        print("Max evaluations (3) reached for this slide to conserve API quota. Falling back to AI Generation.")
                        return None
                        
                    img_id = res['id']
                    if img_id in history:
                        continue
                        
                    img_url = res['urls']['regular']
                    print(f"Evaluating candidate image: {img_url}")
                    temp_path = f"temp_{img_id}.jpg"
                    urllib.request.urlretrieve(img_url, temp_path)
                    
                    time.sleep(5) # Prevent 15 RPM free tier limit!
                    score = validate_image_with_gemini(temp_path, slide_context)
                    evaluations += 1
                    
                    if score >= 75:
                        print(f"ACCEPTED Image {img_id}")
                        history.append(img_id)
                        save_history(history)
                        return temp_path
                    os.remove(temp_path)
        except Exception as e:
            print(f"Search error for query '{query}': {e}")
    return None

@retry(wait=wait_fixed(25), stop=stop_after_attempt(4))
def generate_fallback_image(slide_context):
    print(f"\n--- FALLBACK STAGE: Generating AI Image for '{slide_context}' ---")
    if not API_KEYS: return None
    
    for api_key in API_KEYS:
        client = genai.Client(api_key=api_key)
        try:
            result = client.models.generate_images(
                model='imagen-3.0-generate-001',
                prompt=f"A photorealistic, dark, cinematic, editorial illustration about: {slide_context}. Professional presentation background, lots of negative space at the top.",
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    aspect_ratio="1:1"
                )
            )
            for generated_image in result.generated_images:
                image = Image.open(io.BytesIO(generated_image.image.image_bytes))
                path = f'fallback_{int(time.time())}.jpg'
                image.save(path)
                history = load_history()
                history.append(path.split('.')[0])
                save_history(history)
                return path
        except Exception as e:
            print(f"Fallback Imagen error with key: {e}")
            continue
    return None

def draw_gradient_overlay(width, height):
    """Creates a black gradient from the top down to ensure text is readable."""
    base = Image.new('RGBA', (width, height), (0,0,0,0))
    top = Image.new('RGBA', (width, height), (0,0,0,255))
    mask = Image.new('L', (width, height))
    for y in range(height):
        alpha = int(220 * max(0, 1 - (y / (height * 0.6))))
        for x in range(width):
            mask.putpixel((x, y), alpha)
    base.paste(top, (0, 0), mask)
    return base

def _wrap_text(draw, text, font_bold, font_reg, x_start, x_end):
    """Word-wrap text (with **bold** markers) to fit between x_start and x_end.
    Returns list of line strings."""
    max_width = x_end - x_start
    explicit_lines = text.split('\n')
    final_lines = []
    bold_mode = False
    for e_line in explicit_lines:
        words = e_line.split(' ')
        current_line_words = []
        current_line_width = 0
        for word in words:
            if not word:
                continue
            has_start = word.startswith('**')
            has_end = any(word.endswith(s) for s in ['**', '**.', '**?', '**,', '**!', '**:'])
            clean_word = word.replace('**', '')
            temp_bold = True if has_start else bold_mode
            current_font = font_bold if temp_bold else font_reg
            w_bbox = draw.textbbox((0, 0), clean_word + ' ', font=current_font)
            w = w_bbox[2] - w_bbox[0]
            if current_line_width + w > max_width and current_line_words:
                final_lines.append(' '.join(current_line_words))
                current_line_words = [word]
                current_line_width = w
            else:
                current_line_words.append(word)
                current_line_width += w
            if has_end:
                bold_mode = False
            elif has_start:
                bold_mode = True
        if current_line_words:
            final_lines.append(' '.join(current_line_words))
    return final_lines


def draw_styled_text_lines(draw, text, font_bold, font_reg, start_y,
                           align="center", line_spacing=1.3,
                           width=1080, x_start=90, x_end=None):
    """Draw word-wrapped text with **gold** bold markers.
    Returns the y-coordinate immediately after the last drawn line."""
    if x_end is None:
        x_end = width - 90
    final_lines = _wrap_text(draw, text, font_bold, font_reg, x_start, x_end)
    col_width = x_end - x_start

    ascent, descent = font_bold.getmetrics()
    line_height = ascent + descent
    y = start_y

    for line in final_lines:
        words = line.split(' ')
        total_line_width = 0
        word_metrics = []
        bold_mode = False
        for word in words:
            if not word:
                continue
            has_start = word.startswith('**')
            has_end = any(word.endswith(s) for s in ['**', '**.', '**?', '**,', '**!', '**:'])
            clean_word = word.replace('**', '')
            if has_start:
                bold_mode = True
            current_font = font_bold if bold_mode else font_reg
            current_color = COLOR_GOLD if bold_mode else COLOR_WHITE
            w_bbox = draw.textbbox((0, 0), clean_word + ' ', font=current_font)
            w = w_bbox[2] - w_bbox[0]
            word_metrics.append((clean_word, current_font, current_color, w))
            total_line_width += w
            if has_end:
                bold_mode = False

        if align == "center":
            x = x_start + max(0, (col_width - total_line_width) // 2)
        else:
            x = x_start

        for text_str, font, color, w in word_metrics:
            draw.text((x, y), text_str, font=font, fill=color)
            x += w

        y += int(line_height * line_spacing)

    return y


def draw_bullet_points(draw, bullets, font, start_y,
                       x_start=90, x_end=None, slide_width=1080):
    """Draw bullet list; returns y after last bullet."""
    if x_end is None:
        x_end = slide_width - 90
    bullet_char = '•'
    text_start = x_start + 60
    max_width = x_end - text_start

    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    y = start_y

    for point in bullets:
        words = point.split(' ')
        lines = []
        current_line = []
        current_w = 0
        for word in words:
            w_box = draw.textbbox((0, 0), word + ' ', font=font)
            w_width = w_box[2] - w_box[0]
            if current_w + w_width > max_width and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_w = w_width
            else:
                current_line.append(word)
                current_w += w_width
        if current_line:
            lines.append(' '.join(current_line))

        # Draw bullet only on first line
        draw.text((x_start, y), bullet_char, font=font, fill=COLOR_GOLD)
        for i, line in enumerate(lines):
            draw.text((text_start, y), line, font=font, fill=COLOR_WHITE)
            y += int(line_height * 1.3)
        y += int(line_height * 0.4)   # gap between bullets
    return y

def create_slides(content, slide_image_paths):
    print("\nGenerating slide images with Premium Adaptive Layout...")
    try:
        font_dir = "/usr/share/fonts/truetype/roboto/"
        font_headline_bold = ImageFont.truetype(os.path.join(font_dir, "Roboto-Black.ttf"), 95)
        font_headline_reg  = ImageFont.truetype(os.path.join(font_dir, "Roboto-Bold.ttf"),  95)
        font_stat          = ImageFont.truetype(os.path.join(font_dir, "Roboto-Black.ttf"), 160)
        font_sub           = ImageFont.truetype(os.path.join(font_dir, "Roboto-Medium.ttf"), 48)
        font_quote         = ImageFont.truetype(os.path.join(font_dir, "Roboto-Medium.ttf"), 56)
        font_brand         = ImageFont.truetype(os.path.join(font_dir, "Roboto-Bold.ttf"),   28)
    except:
        print("Using fallback Arial fonts for local test")
        try:
            font_headline_bold = font_headline_reg = font_stat = ImageFont.truetype("arialbd.ttf", 95)
            font_sub = font_quote = ImageFont.truetype("arial.ttf", 48)
            font_brand = ImageFont.truetype("arialbd.ttf", 28)
        except:
            font_headline_bold = font_headline_reg = font_stat = font_sub = font_quote = font_brand = ImageFont.load_default()

    slides_info = content['slides']
    # Cap at 7 slides for safety
    slides_info = slides_info[:7]
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
            print(f"Error loading background {bg_path}: {e}")
            bg = Image.new("RGB", (width, height), (20, 20, 20))

        overlay = draw_gradient_overlay(width, height)
        slide   = Image.alpha_composite(bg.convert("RGBA"), overlay)
        draw    = ImageDraw.Draw(slide)

        # ── Brand bar ──────────────────────────────────────────────
        draw.text((100, 50), "TECH NEWS TODAY", font=font_brand, fill=(255, 255, 255, 180))
        num_text = f"{idx+1:02d} / {len(slides_info):02d}"
        num_w = draw.textbbox((0, 0), num_text, font=font_brand)[2]
        draw.text((width - num_w - 100, 50), num_text, font=font_brand, fill=(255, 255, 255, 180))

        headline_text = slide_info.get("headline", "")
        subtext       = slide_info.get("subtext", "")
        bullets       = slide_info.get("bullet_points", [])
        slide_type    = slide_info.get("slide_type", "context")

        SAFE_BOTTOM = height - 130  # never draw below this line

        # ── COVER layout (slide 1) ──────────────────────────────────
        if slide_type == "cover":
            current_y = 145
            current_y = draw_styled_text_lines(
                draw, headline_text, font_headline_bold, font_headline_reg,
                current_y, align="left", line_spacing=1.25)
            current_y += 28
            draw.rectangle([(90, current_y), (210, current_y + 6)], fill=COLOR_GOLD)
            current_y += 28
            if current_y < SAFE_BOTTOM:
                draw_styled_text_lines(
                    draw, subtext, font_sub, font_sub,
                    current_y, align="left", line_spacing=1.2)

        # ── CONTEXT layout (question/intro) ─────────────────────────
        elif slide_type == "context":
            label = "\u25c6  CONTEXT"
            label_bbox = draw.textbbox((0, 0), label, font=font_brand)
            label_w = label_bbox[2] - label_bbox[0]
            current_y = 200
            draw.text(((width - label_w) // 2, current_y), label, font=font_brand, fill=COLOR_GOLD)
            current_y += label_bbox[3] - label_bbox[1] + 40
            current_y = draw_styled_text_lines(
                draw, headline_text, font_headline_bold, font_headline_reg,
                current_y, align="center", line_spacing=1.25)
            current_y += 42
            if current_y < SAFE_BOTTOM:
                draw_styled_text_lines(
                    draw, subtext, font_sub, font_sub,
                    current_y, align="center", line_spacing=1.2)

        # ── BULLETS layout (detail points) ──────────────────────────
        elif slide_type == "bullets":
            current_y = 145
            current_y = draw_styled_text_lines(
                draw, headline_text, font_headline_bold, font_headline_reg,
                current_y, align="center", line_spacing=1.25)
            current_y += 65
            if current_y < SAFE_BOTTOM:
                draw_bullet_points(
                    draw, bullets if bullets else [subtext],
                    font_sub, current_y)

        # ── STAT layout (big number) ─────────────────────────────────
        elif slide_type == "stat":
            stat_text = headline_text.replace('**', '')
            stat_ascent, stat_descent = font_stat.getmetrics()
            stat_line_h = stat_ascent + stat_descent
            stat_start_y = max(180, (height - stat_line_h) // 2 - 80)
            stat_bbox = draw.textbbox((0, 0), stat_text, font=font_stat)
            stat_w = stat_bbox[2] - stat_bbox[0]
            stat_x = max(90, (width - stat_w) // 2)
            draw.text((stat_x, stat_start_y), stat_text, font=font_stat, fill=COLOR_GOLD)
            current_y = stat_start_y + stat_line_h + 28
            draw.rectangle([(width // 2 - 80, current_y), (width // 2 + 80, current_y + 4)], fill=COLOR_WHITE)
            current_y += 26
            if current_y < SAFE_BOTTOM:
                draw_styled_text_lines(
                    draw, subtext, font_sub, font_sub,
                    current_y, align="center", line_spacing=1.2)

        # ── QUOTE layout (powerful statement) ───────────────────────
        elif slide_type == "quote":
            q_ascent, q_descent = font_stat.getmetrics()
            q_mark_h = q_ascent + q_descent
            current_y = 155
            draw.text((90, current_y), "\u201c", font=font_stat, fill=COLOR_GOLD)
            current_y += q_mark_h + 5
            current_y = draw_styled_text_lines(
                draw, headline_text, font_quote, font_quote,
                current_y, align="center", line_spacing=1.35)
            close_bbox = draw.textbbox((0, 0), "\u201d", font=font_stat)
            close_w = close_bbox[2] - close_bbox[0]
            draw.text((width - 90 - close_w, current_y - 20), "\u201d", font=font_stat, fill=COLOR_GOLD)
            current_y += 26
            draw.rectangle([(width // 2 - 60, current_y), (width // 2 + 60, current_y + 4)], fill=COLOR_GOLD)
            current_y += 22
            if current_y < SAFE_BOTTOM:
                draw_styled_text_lines(
                    draw, subtext, font_sub, font_sub,
                    current_y, align="center", line_spacing=1.2)

        # ── COMPARISON layout (before vs after / old vs new) ─────────
        elif slide_type == "comparison":
            mid_x = width // 2
            current_y = 145
            current_y = draw_styled_text_lines(
                draw, headline_text, font_headline_bold, font_headline_reg,
                current_y, align="center", line_spacing=1.25)
            current_y += 34
            draw.rectangle([(mid_x - 2, current_y), (mid_x + 2, SAFE_BOTTOM)], fill=COLOR_GOLD)
            # Left column
            if len(bullets) >= 1:
                draw.text((90, current_y + 8), "BEFORE", font=font_brand, fill=COLOR_GOLD)
                draw_styled_text_lines(
                    draw, bullets[0], font_sub, font_sub,
                    current_y + 52, align="left",
                    line_spacing=1.25, width=1080,
                    x_start=90, x_end=mid_x - 20)
            # Right column
            if len(bullets) >= 2:
                after_bbox = draw.textbbox((0, 0), "AFTER", font=font_brand)
                after_w = after_bbox[2] - after_bbox[0]
                draw.text((width - 90 - after_w, current_y + 8), "AFTER", font=font_brand, fill=COLOR_GOLD)
                draw_styled_text_lines(
                    draw, bullets[1], font_sub, font_sub,
                    current_y + 52, align="left",
                    line_spacing=1.25, width=1080,
                    x_start=mid_x + 20, x_end=width - 90)

        # ── CTA layout (last slide) ──────────────────────────────────
        elif slide_type == "cta":
            current_y = 200
            draw.rectangle([(width // 2 - 100, current_y - 28), (width // 2 + 100, current_y - 22)], fill=COLOR_GOLD)
            current_y = draw_styled_text_lines(
                draw, headline_text, font_headline_bold, font_headline_reg,
                current_y, align="center", line_spacing=1.25)
            current_y += 42
            if current_y < SAFE_BOTTOM:
                draw_styled_text_lines(
                    draw, subtext, font_sub, font_sub,
                    current_y, align="center", line_spacing=1.2)
            draw.text((90, height - 110), "@techNewsToday  \u2022  Follow for daily AI news",
                      font=font_brand, fill=COLOR_GOLD)

        # ── Fallback (unknown type) ──────────────────────────────────
        else:
            current_y = 190
            current_y = draw_styled_text_lines(
                draw, headline_text, font_headline_bold, font_headline_reg,
                current_y, align="center", line_spacing=1.25)
            current_y += 52
            if current_y < SAFE_BOTTOM:
                draw_styled_text_lines(
                    draw, subtext, font_sub, font_sub,
                    current_y, align="center", line_spacing=1.2)

        out_path = f"slide_{idx+1}.png"
        slide.convert("RGB").save(out_path)
        final_slide_paths.append(out_path)
        print(f"  ✓ Slide {idx+1} [{slide_type}] saved → {out_path}")

    return final_slide_paths

def upload_image(file_path):
    print(f"Uploading {file_path} to freeimage.host...")
    url = "https://freeimage.host/api/1/upload"
    with open(file_path, "rb") as f: b64 = base64.b64encode(f.read()).decode("utf-8")
    data = urllib.parse.urlencode({"key": "6d207e02198a847aa98d0a2a901485a5", "action": "upload", "source": b64, "format": "json"}).encode("utf-8")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data)) as res:
            return json.loads(res.read().decode())["image"]["url"]
    except: return None

def post_to_instagram(image_urls, caption):
    print("Posting to Instagram...")
    item_ids = []
    for url in image_urls:
        req_url = f"https://graph.instagram.com/v20.0/{IG_ACCOUNT_ID}/media"
        data = urllib.parse.urlencode({'image_url': url, 'is_carousel_item': 'true', 'access_token': IG_ACCESS_TOKEN}).encode('utf-8')
        try:
            with urllib.request.urlopen(urllib.request.Request(req_url, data=data)) as res:
                item_ids.append(json.loads(res.read().decode())['id'])
        except: return False
        time.sleep(2)
        
    req_url = f"https://graph.instagram.com/v20.0/{IG_ACCOUNT_ID}/media"
    data = urllib.parse.urlencode({'media_type': 'CAROUSEL', 'children': ','.join(item_ids), 'caption': caption, 'access_token': IG_ACCESS_TOKEN}).encode('utf-8')
    try:
        with urllib.request.urlopen(urllib.request.Request(req_url, data=data)) as res:
            carousel_id = json.loads(res.read().decode())['id']
    except: return False
        
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
            print(f"SUCCESS! Published post ID: {json.loads(res.read().decode())['id']}")
            return True
    except: return False

if __name__ == "__main__":
    try:
        news_text = get_latest_news()
        content = generate_post_content(news_text)
        print(f"Generated Content: {json.dumps(content, indent=2)}")
        
        slide_image_paths = []
        for i, slide in enumerate(content['slides']):
            print(f"\\n--- Processing Background for Slide {i+1} ---")
            img_path = get_valid_unsplash_image(slide['search_queries'], slide['headline'])
            if not img_path:
                img_path = generate_fallback_image(slide['headline'])
            if not img_path:
                print("Using empty black fallback image.")
                img_path = "fallback_black.jpg"
                Image.new("RGB", (1080, 1080), (20, 20, 20)).save(img_path)
            slide_image_paths.append(img_path)
            
        final_slides = create_slides(content, slide_image_paths)
        
        urls = []
        for slide in final_slides:
            url = upload_image(slide)
            if url: urls.append(url)
            
        if len(urls) == len(final_slides):
            success = post_to_instagram(urls, content['caption'])
            if success:
                # Save the article title so we never post it again
                news_history = load_news_history()
                news_history.append(content.get('original_title', '').strip())
                save_news_history(news_history)
        else:
            print("Failed to upload all images.")
            sys.exit(1)
    except Exception as e:
        print(f"Critical error in main pipeline: {e}")
        sys.exit(1)
