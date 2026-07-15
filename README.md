# Instagram Tech Bot

A fully‑automated GitHub Action that pulls the latest tech news, generates carousel slides with Gemini, finds matching images from Unsplash, and posts the result to Instagram.

## How it works
1. **Trigger** – Create a GitHub Issue titled `post` (or use the Slack `/github open` integration). The workflow runs automatically.
2. **Fetch news** – The script pulls the latest TechCrunch RSS feed, filters out any titles that already exist in `used_news.txt` (so the bot never repeats a topic).
3. **Generate content** – Gemini (Gemini‑2.5‑flash) receives the fresh headlines and produces a JSON payload describing the overall topic, slide headlines, sub‑text, bullet points and Unsplash search queries.
4. **Image creation** – For each slide the bot searches Unsplash, downloads up to three images, and assembles them into Instagram‑ready carousel slides.
5. **Post to Instagram** – The slides and a caption are uploaded via the Instagram Graph API.
6. **Persist history** – After a successful post the chosen article title is appended to `used_news.txt` and the used Unsplash image URLs are stored in `used_images.txt`. Both files are committed back to the repository so the bot remembers what it has already published.

## Configuration & Secrets
| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Your Gemini API key (store as a GitHub secret) |
| `UNSPLASH_ACCESS_KEY` | Unsplash API key (secret) |
| `IG_ACCESS_TOKEN` | Instagram Graph API token (secret) |
| `IG_ACCOUNT_ID` | Instagram Business Account ID (secret) |
| `used_images.txt` / `used_news.txt` | Auto‑generated files that track previously used images and news titles |

Add these as repository **Secrets** (`Settings → Secrets and variables → Actions`).

## Running locally (for testing)
```bash
# Install dependencies (Python 3.10+)
pip install -r requirements.txt

# Export your keys locally
export GEMINI_API_KEY=...   # Windows: set GEMINI_API_KEY=...
export UNSPLASH_ACCESS_KEY=...
export IG_ACCESS_TOKEN=...
export IG_ACCOUNT_ID=...

# Execute the script manually
python main.py
```

## Updating the bot
Whenever you make changes (e.g., adjust the prompt, modify the word‑wrapping logic, or tweak the news‑tracking behaviour) simply commit and push – the GitHub workflow will pick up the latest version on the next trigger.

---
