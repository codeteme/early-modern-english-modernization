import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path


# --- Path helpers ---
SRC_DIR = Path(__file__).resolve().parent              # .../src
PROJECT_ROOT = SRC_DIR.parent                         # parent of src and data
DATA_DIR = PROJECT_ROOT / "data"                      # .../data
TEXTS_DIR = DATA_DIR / "texts"                        # .../data/texts


def scrape_shakespeare_columns(work_id, output_dir=TEXTS_DIR):
    """
    Scrape Shakespeare text with two columns (First Folio and Modern text)
    Saves to old_text.txt and new_text.txt in:
      data/texts/<PlayTitle>_<work_id>/
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://www.shakespeareswords.com/Public/Play.aspx?WorkId={work_id}"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        title_elem = soup.find('title')
        title = title_elem.get_text().strip() if title_elem else f"Work_{work_id}"

        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_')

        # Extract First Folio text (old text)
        old_text_lines = []
        first_folio_cells = soup.find_all('td', class_='firstFolioTd')
        for cell in first_folio_cells:
            text = cell.get_text(separator=' ', strip=True)
            if text:
                old_text_lines.append(text)

        # Extract Modern text (new text)
        new_text_lines = []
        modern_cells = soup.find_all('td', class_='penguinTextFF')
        for cell in modern_cells:
            text = cell.get_text(separator=' ', strip=True)
            if text:
                new_text_lines.append(text)

        if not old_text_lines and not new_text_lines:
            print(f"✗ No text content found for WorkId={work_id}")
            return None

        # ✅ FIX: make folder unique per work_id so files don't overwrite
        work_dir = output_dir / f"{safe_title}_{work_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

        old_path = work_dir / "old_text.txt"
        with open(old_path, 'w', encoding='utf-8') as f:
            f.write(f"Title: {title}\n")
            f.write("Version: First Folio\n")
            f.write(f"Source: {url}\n")
            f.write("=" * 70 + "\n\n")
            f.write("\n".join(old_text_lines))
        print(f"✓ Saved First Folio text: {old_path}")

        new_path = work_dir / "new_text.txt"
        with open(new_path, 'w', encoding='utf-8') as f:
            f.write(f"Title: {title}\n")
            f.write("Version: Modern Text\n")
            f.write(f"Source: {url}\n")
            f.write("=" * 70 + "\n\n")
            f.write("\n".join(new_text_lines))
        print(f"✓ Saved Modern text: {new_path}")

        return {
            "old_text": str(old_path),
            "new_text": str(new_path),
            "old_lines": len(old_text_lines),
            "new_lines": len(new_text_lines),
        }

    except Exception as e:
        print(f"✗ Error scraping WorkId={work_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def merge_txt_files_by_prefix(search_dir, prefix, output_path):
    """
    Find all .txt files under search_dir whose filename starts with prefix,
    concatenate them, and save to output_path.
    """
    search_dir = Path(search_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in search_dir.rglob("*.txt") if p.name.startswith(prefix)],
        key=lambda p: str(p).lower()
    )

    if not files:
        print(f"✗ No files found with prefix '{prefix}' in {search_dir}")
        return None

    merged_chunks = []
    for p in files:
        try:
            merged_chunks.append(
                f"\n\n{'='*70}\nFILE: {p.relative_to(PROJECT_ROOT)}\n{'='*70}\n"
            )
            merged_chunks.append(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  Warning: could not read {p}: {e}")

    output_path.write_text("".join(merged_chunks), encoding="utf-8")
    print(f"✓ Merged {len(files)} '{prefix}*.txt' files -> {output_path}")
    return str(output_path)


def scrape_multiple_works(work_ids, output_dir=TEXTS_DIR, delay=2, merge_prefixes=True):
    """
    Scrape multiple works with rate limiting.
    After scraping, merges:
      - all "new*.txt" into data/new_merged.txt
      - all "old*.txt" into data/old_merged.txt
    """
    output_dir = Path(output_dir)
    results = {}

    for work_id in work_ids:
        print(f"\n{'='*50}")
        print(f"Processing WorkId={work_id}...")
        print('='*50)
        results[work_id] = scrape_shakespeare_columns(work_id, output_dir)

        if work_id != work_ids[-1]:
            time.sleep(delay)

    if merge_prefixes:
        print(f"\n{'='*50}")
        print("Merging all prefix-matching txt files...")
        print('='*50)

        merge_txt_files_by_prefix(
            search_dir=output_dir,
            prefix="new",
            output_path=DATA_DIR / "new_merged.txt"
        )
        merge_txt_files_by_prefix(
            search_dir=output_dir,
            prefix="old",
            output_path=DATA_DIR / "old_merged.txt"
        )

    return results


if __name__ == "__main__":
    work_ids = [1,2,3,4,5,6,7,8,9,10]
    scrape_multiple_works(work_ids, merge_prefixes=True)
