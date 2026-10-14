import requests
import certifi
import re

url = "https://investor.nvidia.com/financial-info/annual-reports-and-proxies/default.aspx"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

res = requests.get(url, headers=headers, verify=certifi.where(), timeout=15)
# Find anchor tags with text and href
pattern = r'<a[^>]+href="([^"]+\.pdf)"[^>]*>(.*?)</a>'
matches = re.findall(pattern, res.text, re.DOTALL | re.IGNORECASE)

print(f"Total PDF links found: {len(matches)}")
for href, text in matches:
    clean_text = re.sub(r'<[^>]+>', '', text).strip()
    clean_href = "https:" + href if href.startswith("//") else href
    print(f"{clean_text} -> {clean_href}")

