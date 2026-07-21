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
    """Full-slide dark gradient using pure PIL (no numpy dependency)."""
    base = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw_g = ImageDraw.Draw(base)
    for y in range(height):
        t = y / height
        alpha = max(120, int(210 - t * 70))
        draw_g.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    return base


def _wrap_text(draw, text, font_bold, font_reg, x_start, x_end):
    """Word-wrap text (with **bold** markers) to fit between x_start and x_end.
    Returns list of line strings."""
    max_width = x_end - x_start
    final_lines = []
    bold_mode = False
    for e_line in text.split('\n'):
        current_line_words = []
        current_line_width = 0
        for word in e_line.split(' '):
            if not word:
                continue
            has_start = word.startswith('**')
            has_end = any(word.endswith(s) for s in ['**', '**.', '**?', '**,', '**!', '**:'])
            clean_word = word.replace('**', '')
            temp_bold = True if has_start else bold_mode
            current_font = font_bold if temp_bold else font_reg
            w = draw.textbbox((0, 0), clean_word + ' ', font=current_font)[2]
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


def _line_height(font):
    a, d = font.getmetrics()
    return a + d


def draw_styled_text_lines(draw, text, font_bold, font_reg, start_y,
                           align="center", line_spacing=1.3,
                           width=1080, x_start=90, x_end=None):
    """Draw word-wrapped **gold** bold text. Returns y after last line."""
    if x_end is None:
        x_end = width - 90
    lines = _wrap_text(draw, text, font_bold, font_reg, x_start, x_end)
    col_w = x_end - x_start
    lh = _line_height(font_bold)
    y = start_y

    for line in lines:
        total_w = 0
        word_metrics = []
        bold_mode = False
        for word in line.split(' '):
            if not word:
                continue
            has_start = word.startswith('**')
            has_end = any(word.endswith(s) for s in ['**', '**.', '**?', '**,', '**!', '**:'])
            clean = word.replace('**', '')
            if has_start:
                bold_mode = True
            font = font_bold if bold_mode else font_reg
            color = COLOR_GOLD if bold_mode else COLOR_WHITE
            w = draw.textbbox((0, 0), clean + ' ', font=font)[2]
            word_metrics.append((clean, font, color, w))
            total_w += w
            if has_end:
                bold_mode = False

        x = (x_start + max(0, (col_w - total_w) // 2)) if align == "center" else x_start
        for txt, font, color, w in word_metrics:
            draw.text((x, y), txt, font=font, fill=color)
            x += w
        y += int(lh * line_spacing)

    return y


def draw_plain_text_wrapped(draw, text, font, start_y, color,
                            align="center", line_spacing=1.25,
                            width=1080, x_start=90, x_end=None):
    """Draw plain (non-bold-markup) wrapped text. Returns y after last line."""
    if x_end is None:
        x_end = width - 90
    col_w = x_end - x_start
    lh = _line_height(font)
    y = start_y

    # Word wrap
    lines = []
    current_words = []
    current_w = 0
    for word in text.split():
        w = draw.textbbox((0, 0), word + ' ', font=font)[2]
        if current_w + w > col_w and current_words:
            lines.append(' '.join(current_words))
            current_words = [word]
            current_w = w
        else:
            current_words.append(word)
            current_w += w
    if current_words:
        lines.append(' '.join(current_words))

    for line in lines:
        line_w = draw.textbbox((0, 0), line, font=font)[2]
        x = (x_start + max(0, (col_w - line_w) // 2)) if align == "center" else x_start
        draw.text((x, y), line, font=font, fill=color)
        y += int(lh * line_spacing)

    return y


def draw_bullet_points(draw, bullets, font, start_y,
                       x_start=90, x_end=None, slide_width=1080):
    """Draw bullet list; returns y after last bullet."""
    if x_end is None:
        x_end = slide_width - 90
    text_start = x_start + 52
    max_w = x_end - text_start
    lh = _line_height(font)
    y = start_y

    for point in bullets:
        # Wrap point text
        lines = []
        current = []
        cw = 0
        for word in point.split():
            w = draw.textbbox((0, 0), word + ' ', font=font)[2]
            if cw + w > max_w and current:
                lines.append(' '.join(current))
                current = [word]
                cw = w
            else:
                current.append(word)
                cw += w
        if current:
            lines.append(' '.join(current))

        draw.text((x_start + 8, y + 4), '•', font=font, fill=COLOR_GOLD)
        for line in lines:
            draw.text((text_start, y), line, font=font, fill=COLOR_WHITE)
            y += int(lh * 1.3)
        y += int(lh * 0.35)

    return y


def _auto_font(font_large, font_medium, font_small, draw, text, x_start, x_end, max_lines=3):
    """Pick the largest font where text fits within max_lines."""
    clean_text = text.replace('**', '')
    for font in (font_large, font_medium, font_small):
        lines = _wrap_text(draw, clean_text, font, font, x_start, x_end)
        if len(lines) <= max_lines:
            return font
    return font_small


def create_slides(content, slide_image_paths):
    print("\nGenerating slide images with Premium Adaptive Layout...")
    # ── Font loading ──────────────────────────────────────────────────
    try:
        fd = "/usr/share/fonts/truetype/roboto/"
        fB  = lambda s: ImageFont.truetype(fd + "Roboto-Black.ttf",  s)
        fBo = lambda s: ImageFont.truetype(fd + "Roboto-Bold.ttf",   s)
        fM  = lambda s: ImageFont.truetype(fd + "Roboto-Medium.ttf", s)
        font_hl_xl   = fB(88);  font_hl_xl_r  = fBo(88)
        font_hl_lg   = fB(72);  font_hl_lg_r  = fBo(72)
        font_hl_md   = fB(58);  font_hl_md_r  = fBo(58)
        font_stat    = fB(150)
        font_sub_lg  = fM(46)
        font_sub_md  = fM(40)
        font_quote   = fM(52)
        font_brand   = fBo(26)
    except Exception as ex:
        print(f"Font load error: {ex} — using fallback")
        try:
            fB = fBo = lambda s: ImageFont.truetype("arialbd.ttf", s)
            fM       = lambda s: ImageFont.truetype("arial.ttf",   s)
            font_hl_xl = font_hl_xl_r = fB(88)
            font_hl_lg = font_hl_lg_r = fB(72)
            font_hl_md = font_hl_md_r = fB(58)
            font_stat  = fB(150)
            font_sub_lg = fM(46); font_sub_md = fM(40)
            font_quote  = fM(52); font_brand  = fB(26)
        except:
            _f = ImageFont.load_default()
            font_hl_xl = font_hl_xl_r = font_hl_lg = font_hl_lg_r = \
            font_hl_md = font_hl_md_r = font_stat = \
            font_sub_lg = font_sub_md = font_quote = font_brand = _f

    slides_info = content['slides'][:7]
    final_slide_paths = []
    W, H = 1080, 1080
    MARGIN   = 90       # left/right margin
    TOP_PAD  = 80       # space at top for counter
    BOT_PAD  = 80       # safe zone at bottom
    CONTENT_TOP    = TOP_PAD + 30
    CONTENT_BOTTOM = H - BOT_PAD
    CONTENT_H      = CONTENT_BOTTOM - CONTENT_TOP

    for idx, slide_info in enumerate(slides_info):
        # ── Background ───────────────────────────────────────────────
        bg_path = slide_image_paths[idx]
        try:
            base_bg = Image.open(bg_path).convert("RGB")
            bw, bh = base_bg.size
            md = min(bw, bh)
            bg = base_bg.crop(((bw-md)//2, (bh-md)//2, (bw+md)//2, (bh+md)//2))
            bg = bg.resize((W, H), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"  BG error {bg_path}: {e}")
            bg = Image.new("RGB", (W, H), (18, 18, 22))

        # Dark overlay for readability
        dark = Image.new("RGBA", (W, H), (0, 0, 0, 185))
        slide = Image.alpha_composite(bg.convert("RGBA"), dark)
        draw  = ImageDraw.Draw(slide)

        # ── Slide counter — top right only, no other branding ────────
        num_text = f"{idx+1:02d} / {len(slides_info):02d}"
        nw = draw.textbbox((0, 0), num_text, font=font_brand)[2]
        draw.text((W - MARGIN - nw, 36), num_text,
                  font=font_brand, fill=(255, 255, 255, 130))

        headline_text = slide_info.get("headline", "").strip()
        subtext       = slide_info.get("subtext",  "").strip()
        bullets       = slide_info.get("bullet_points", [])
        slide_type    = slide_info.get("slide_type", "context")

        # Auto-scale headline font so it always fits in ≤3 lines
        hl_bold = _auto_font(font_hl_xl, font_hl_lg, font_hl_md,
                             draw, headline_text, MARGIN, W - MARGIN, max_lines=3)
        if hl_bold is font_hl_xl:   hl_reg = font_hl_xl_r
        elif hl_bold is font_hl_lg: hl_reg = font_hl_lg_r
        else:                        hl_reg = font_hl_md_r

        lh_hl  = _line_height(hl_bold)
        hl_lines  = _wrap_text(draw, headline_text, hl_bold, hl_reg, MARGIN, W - MARGIN)
        hl_block_h = int(lh_hl * 1.3 * len(hl_lines))
        lh_sub    = _line_height(font_sub_lg)

        # ── COVER — left-aligned ──────────────────────────────────────
        if slide_type == "cover":
            sub_lines = _wrap_text(draw, subtext, font_sub_lg, font_sub_lg, MARGIN, W - MARGIN)
            sub_h     = int(lh_sub * 1.25 * len(sub_lines)) if sub_lines else 0
            total_h   = hl_block_h + 28 + 6 + 22 + sub_h
            start_y   = CONTENT_TOP + max(0, (CONTENT_H - total_h) // 2)

            ey = draw_styled_text_lines(
                draw, headline_text, hl_bold, hl_reg,
                start_y, align="left", line_spacing=1.3,
                x_start=MARGIN, x_end=W - MARGIN)
            ey += 24
            draw.rectangle([(MARGIN, ey), (MARGIN + 120, ey + 5)], fill=COLOR_GOLD)
            ey += 22
            draw_plain_text_wrapped(draw, subtext, font_sub_lg, ey,
                                    COLOR_WHITE, align="left",
                                    line_spacing=1.25,
                                    x_start=MARGIN, x_end=W - MARGIN)

        # ── CONTEXT — centered ───────────────────────────────────────
        elif slide_type == "context":
            sub_lines = _wrap_text(draw, subtext, font_sub_lg, font_sub_lg, MARGIN, W - MARGIN)
            sub_h     = int(lh_sub * 1.25 * len(sub_lines)) if sub_lines else 0
            total_h   = hl_block_h + 36 + sub_h
            start_y   = CONTENT_TOP + max(0, (CONTENT_H - total_h) // 2)

            ey = draw_styled_text_lines(
                draw, headline_text, hl_bold, hl_reg,
                start_y, align="center", line_spacing=1.3,
                x_start=MARGIN, x_end=W - MARGIN)
            draw.rectangle([(W//2 - 50, ey + 10), (W//2 + 50, ey + 13)],
                           fill=(255, 255, 255, 60))
            ey += 36
            draw_plain_text_wrapped(draw, subtext, font_sub_lg, ey,
                                    COLOR_WHITE, align="center",
                                    line_spacing=1.25,
                                    x_start=MARGIN, x_end=W - MARGIN)

        # ── BULLETS — center headline, left bullets ──────────────────
        elif slide_type == "bullets":
            bullet_list = bullets if bullets else [subtext]
            lh_b  = _line_height(font_sub_lg)
            bul_h = sum(int(lh_b * 1.3) + int(lh_b * 0.35) for _ in bullet_list)
            total_h = hl_block_h + 50 + bul_h
            start_y = CONTENT_TOP + max(0, (CONTENT_H - total_h) // 2)

            ey = draw_styled_text_lines(
                draw, headline_text, hl_bold, hl_reg,
                start_y, align="center", line_spacing=1.3,
                x_start=MARGIN, x_end=W - MARGIN)
            draw.rectangle([(MARGIN, ey + 12), (W - MARGIN, ey + 14)],
                           fill=(255, 255, 255, 35))
            draw_bullet_points(draw, bullet_list, font_sub_lg, ey + 44,
                               x_start=MARGIN, x_end=W - MARGIN)

        # ── STAT — left-aligned ──────────────────────────────────────
        elif slide_type == "stat":
            stat_text = headline_text.replace('**', '')
            # Auto-scale the big number font
            stat_font = font_stat
            for sz in (130, 110, 90, 70):
                try:
                    sf = fB(sz)
                    if draw.textbbox((0, 0), stat_text, font=sf)[2] <= W - 2 * MARGIN:
                        stat_font = sf
                        break
                except:
                    pass
            stat_lh = _line_height(stat_font)
            sub_lines = _wrap_text(draw, subtext, font_sub_lg, font_sub_lg, MARGIN, W - MARGIN)
            sub_h     = int(lh_sub * 1.25 * len(sub_lines)) if sub_lines else 0
            # Context line above number + number + line + description
            ctx_lh    = _line_height(font_sub_lg)
            total_h   = ctx_lh + 20 + stat_lh + 24 + 5 + 22 + sub_h
            start_y   = CONTENT_TOP + max(0, (CONTENT_H - total_h) // 2)

            # Small context text above the number — left aligned
            draw_plain_text_wrapped(draw, "Key figure", font_sub_md, start_y,
                                    (180, 180, 180, 255), align="left",
                                    line_spacing=1.0,
                                    x_start=MARGIN, x_end=W - MARGIN)
            num_y = start_y + ctx_lh + 20
            draw.text((MARGIN, num_y), stat_text, font=stat_font, fill=COLOR_GOLD)
            ey = num_y + stat_lh + 20
            draw.rectangle([(MARGIN, ey), (MARGIN + 120, ey + 5)], fill=COLOR_GOLD)
            ey += 22
            draw_plain_text_wrapped(draw, subtext, font_sub_lg, ey,
                                    COLOR_WHITE, align="left",
                                    line_spacing=1.25,
                                    x_start=MARGIN, x_end=W - MARGIN)

        # ── QUOTE — centered, no big quotation marks ─────────────────
        elif slide_type == "quote":
            q_lh     = _line_height(font_quote)
            q_lines  = _wrap_text(draw, headline_text, font_quote, font_quote,
                                  MARGIN + 30, W - MARGIN - 30)
            q_h      = int(q_lh * 1.35 * len(q_lines))
            attr_lh  = _line_height(font_sub_md)
            total_h  = q_h + 30 + 4 + 20 + attr_lh
            start_y  = CONTENT_TOP + max(0, (CONTENT_H - total_h) // 2)

            ey = draw_styled_text_lines(
                draw, headline_text, font_quote, font_quote,
                start_y, align="center", line_spacing=1.35,
                x_start=MARGIN + 30, x_end=W - MARGIN - 30)
            ey += 26
            draw.rectangle([(W//2 - 60, ey), (W//2 + 60, ey + 4)], fill=COLOR_GOLD)
            ey += 22
            # Attribution — subtext holds "— Name, Title"
            attr = subtext if subtext else ""
            draw_plain_text_wrapped(draw, attr, font_sub_md, ey,
                                    (180, 180, 180, 255), align="center",
                                    line_spacing=1.2,
                                    x_start=MARGIN, x_end=W - MARGIN)

        # ── COMPARISON — two columns ──────────────────────────────────
        elif slide_type == "comparison":
            mid   = W // 2
            cx0, cx1 = MARGIN, mid - 20
            cx2, cx3 = mid + 20, W - MARGIN
            bl0  = bullets[0] if bullets else ""
            bl1  = bullets[1] if len(bullets) > 1 else ""
            l0   = _wrap_text(draw, bl0, font_sub_md, font_sub_md, cx0, cx1)
            l1   = _wrap_text(draw, bl1, font_sub_md, font_sub_md, cx2, cx3)
            col_h = int(_line_height(font_sub_md) * 1.3 * max(len(l0), len(l1), 1))
            lh_lbl = _line_height(font_brand)
            total_h = hl_block_h + 34 + lh_lbl + 16 + col_h
            start_y = CONTENT_TOP + max(0, (CONTENT_H - total_h) // 2)

            ey = draw_styled_text_lines(
                draw, headline_text, hl_bold, hl_reg,
                start_y, align="center", line_spacing=1.3,
                x_start=MARGIN, x_end=W - MARGIN)
            ey += 30
            draw.rectangle([(mid - 2, ey), (mid + 2, CONTENT_BOTTOM)], fill=COLOR_GOLD)
            # Column labels
            draw.text((cx0, ey + 6), "BEFORE", font=font_brand, fill=COLOR_GOLD)
            aw = draw.textbbox((0, 0), "AFTER", font=font_brand)[2]
            draw.text((cx3 - aw, ey + 6), "AFTER", font=font_brand, fill=COLOR_GOLD)
            col_y = ey + lh_lbl + 18
            draw_plain_text_wrapped(draw, bl0, font_sub_md, col_y,
                                    COLOR_WHITE, align="left",
                                    x_start=cx0, x_end=cx1)
            draw_plain_text_wrapped(draw, bl1, font_sub_md, col_y,
                                    COLOR_WHITE, align="left",
                                    x_start=cx2, x_end=cx3)

        # ── CTA — two gold lines framing centered text ────────────────
        elif slide_type == "cta":
            sub_lines = _wrap_text(draw, subtext, font_sub_lg, font_sub_lg, MARGIN, W - MARGIN)
            sub_h     = int(lh_sub * 1.25 * len(sub_lines)) if sub_lines else 0
            total_h   = 6 + 28 + hl_block_h + 28 + sub_h + 28 + 6
            start_y   = CONTENT_TOP + max(0, (CONTENT_H - total_h) // 2)

            # Top gold line
            draw.rectangle([(W//2 - 110, start_y), (W//2 + 110, start_y + 5)], fill=COLOR_GOLD)
            ey = start_y + 30
            ey = draw_styled_text_lines(
                draw, headline_text, hl_bold, hl_reg,
                ey, align="center", line_spacing=1.3,
                x_start=MARGIN, x_end=W - MARGIN)
            ey += 26
            draw_plain_text_wrapped(draw, subtext, font_sub_lg, ey,
                                    COLOR_WHITE, align="center",
                                    line_spacing=1.25,
                                    x_start=MARGIN, x_end=W - MARGIN)
            ey += sub_h + 26
            # Bottom gold line
            draw.rectangle([(W//2 - 110, ey), (W//2 + 110, ey + 5)], fill=COLOR_GOLD)

        # ── Fallback ─────────────────────────────────────────────────
        else:
            sub_lines = _wrap_text(draw, subtext, font_sub_lg, font_sub_lg, MARGIN, W - MARGIN)
            sub_h     = int(lh_sub * 1.25 * len(sub_lines)) if sub_lines else 0
            total_h   = hl_block_h + 40 + sub_h
            start_y   = CONTENT_TOP + max(0, (CONTENT_H - total_h) // 2)
            ey = draw_styled_text_lines(
                draw, headline_text, hl_bold, hl_reg,
                start_y, align="left", line_spacing=1.3,
                x_start=MARGIN, x_end=W - MARGIN)
            ey += 36
            draw_plain_text_wrapped(draw, subtext, font_sub_lg, ey,
                                    COLOR_WHITE, align="left",
                                    line_spacing=1.25,
                                    x_start=MARGIN, x_end=W - MARGIN)

        out_path = f"slide_{idx+1}.png"
        slide.convert("RGB").save(out_path)
        final_slide_paths.append(out_path)
        print(f"  \u2713 Slide {idx+1} [{slide_type}] \u2192 {out_path}")

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
