import os
import time
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "saved_pages")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add all your 35+ pages here
PAGES_TO_SCRAPE = {
    #"about_us.txt": "https://www.rgukt.ac.in/about-introduction.html",
   # "vision_mission.txt": "https://www.rgukt.ac.in/vision-mission.html",
   # "naac.txt": "https://www.rgukt.ac.in/naac.html",
   # "nirf.txt": "https://www.rgukt.ac.in/nirf.html",
   # "vc.txt": "https://www.rgukt.ac.in/vc.html",
   # "vc_succession.txt": "https://www.rgukt.ac.in/vc_succession.html",
   # "administration.txt": "https://www.rgukt.ac.in/administration-section.html",
   # "cd.txt": "https://www.rgukt.ac.in/cd.html",
    # "acdcprgm.txt": "https://www.rgukt.ac.in/academicprogrammes.html",
     #"curricula.txt":"https://www.rgukt.ac.in/curricula.html",
     #"acdccld.txt":"https://www.rgukt.ac.in/academiccalender.html",
#    "departments.txt": "https://www.rgukt.ac.in/departments.html",
    "examination.txt":"https://www.rgukt.ac.in/examination.html",
#    "tmtable.txt":"https://www.rgukt.ac.in/timetable.html",
#    "bio.txt":"https://www.rgukt.ac.in/department/bio-sciences",
#    "cse.txt":"https://www.rgukt.ac.in/department/cse",
 #   "ce.txt":"https://www.rgukt.ac.in/department/ce",
  #  "che.txt":"https://www.rgukt.ac.in/department/che",
   # "eee.txt":"https://www.rgukt.ac.in/department/eee",
    #"ece.txt":"https://www.rgukt.ac.in/department/ece",
#    "hss.txt":"https://www.rgukt.ac.in/department/hss",
 #   "it.txt":"https://www.rgukt.ac.in/department/it",
  #  "me.txt":"https://www.rgukt.ac.in/department/me",
   # "mme.txt":"https://www.rgukt.ac.in/department/mme",
#    "maths.txt":"https://www.rgukt.ac.in/department/maths",
#    "physics.txt":"https://www.rgukt.ac.in/department/physics",
#    "schoolmng.txt":"https://www.rgukt.ac.in/department/schoolmng",
 #   "hostels.txt":"https://www.rgukt.ac.in/hostels.html",
  #  "ch.txt":"https://www.rgukt.ac.in/conference-halls.html",
#    "counseling.txt":"https://www.rgukt.ac.in/counseling.html",
#    "library.txt":"https://www.rgukt.ac.in/library",
#    "medical.txt":"https://www.rgukt.ac.in/hospital.html",
#    "bank.txt":"https://www.rgukt.ac.in/bank-atm.html",
#   "post.txt":"https://www.rgukt.ac.in/post-office.html",
 #   "sc.txt":"https://www.rgukt.ac.in/shopping-complex.html",
#    "gym.txt":"https://www.rgukt.ac.in/gym.html",
 #   "sports.txt":"https://www.rgukt.ac.in/sports.html",
#    "placements.txt":"https://www.rgukt.ac.in/placements.html",
 #   "careers.txt":"https://www.rgukt.ac.in/careers.html",
  #  "alumni.txt":"https://www.rgukt.ac.in/alumni.html",
#    "cus.txt":"https://www.rgukt.ac.in/contactus.html",
 #   "alumni_network.txt": "https://www.alumni.rgukt.ac.in/",
    "departments_faculty.txt":"https://www.rgukt.ac.in/faculty-directory?dept="
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        element.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def scrape_all():
    total_pages = len(PAGES_TO_SCRAPE)
    print("=" * 65)
    print(f"  RGUKT Basar Knowledge Harvester (Total Pages: {total_pages})")
    print(f"  Destination: {OUTPUT_DIR}")
    print("=" * 65)

    session = requests.Session()
    session.trust_env = False

    success_count = 0

    for idx, (filename, url) in enumerate(PAGES_TO_SCRAPE.items(), start=1):
        if "/api/" in url:
            print(f"[{idx}/{total_pages}] Skipping disallowed URL: {url}")
            continue

        filepath = os.path.join(OUTPUT_DIR, filename)
        print(f"[{idx}/{total_pages}] Fetching: {url} ...")

        try:
            response = session.get(url, headers=HEADERS, timeout=15)
            # Ensure proper text encoding
            response.encoding = response.apparent_encoding or "utf-8"

            if response.status_code == 200:
                clean_text = clean_html(response.text)

                if len(clean_text) < 50:
                    print(f"  ⚠️ Warning: {filename} had almost no text. Check if URL is valid.")
                
                header = f"Source URL: {url}\nOfficial Ref: RGUKT-B/Scraped\n{'=' * 65}\n\n"
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(header + clean_text)

                print(f"  ✅ Saved {filename} ({len(clean_text)} characters)")
                success_count += 1
            else:
                print(f"  ❌ HTTP {response.status_code} on {url}")

        except requests.exceptions.RequestException as e:
            print(f"  ❌ Failed to reach {url}: {e}")

        # 1.5s delay to be polite to the university infrastructure
        time.sleep(1.5)

    print("\n" + "=" * 65)
    print(f"Scraping completed: {success_count}/{total_pages} files saved.")
    print("=" * 65)

if __name__ == "__main__":
    scrape_all()
