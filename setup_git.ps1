git init
git add .
git commit -m "Initial commit of IG automation bot"
gh repo create instagram-tech-bot --private --source=. --remote=origin --push
gh secret set GEMINI_API_KEY -b"AQ.Ab8RN6Ktp6jL1CbmhTpbPu3i8bhVsx5SW-KEJ5wH0fv3AVevuQ"
gh secret set UNSPLASH_ACCESS_KEY -b"NWQiPI969pnVNQrCe0r3hgqNjqOZTZPdmYxu-TTAw5I"
gh secret set IG_ACCESS_TOKEN -b"IGAASp0s2MxZApBZAFlRVnRyakxUdWVUblhqYUJubUVhcklPZA01JMTU5dFdva3BEUUt5cExoaUZAKbHlnLXlWVE8xQ0NNWi1KeXY5bGRfejA2LWF2LWJKS29DZAk11RnBWdGFHM2k3aVoxZAW9lel93SUVKSWVycGhGNTJ2bFQyMkJRUQZDZD"
gh secret set IG_ACCOUNT_ID -b"17841474159027766"
