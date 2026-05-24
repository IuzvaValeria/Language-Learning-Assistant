from pathlib import Path
import json
import xml.etree.ElementTree as ET

RAW_FILE = Path("data/raw/jmdict/JMdict_e.xml")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "jmdict_words.jsonl"

def parse_jmdict():
    words = []

    if not RAW_FILE.exists():
        print(f"File not found: {RAW_FILE}")
        return words

    for event, elem in ET.iterparse(RAW_FILE, events=("end",)):
        if elem.tag != "entry":
            continue

        keb = elem.find("k_ele/keb")
        reb = elem.find("r_ele/reb")
        glosses = elem.findall("sense/gloss")

        if reb is None or not glosses:
            elem.clear()
            continue

        word = keb.text if keb is not None else reb.text
        reading = reb.text
        meanings = [g.text for g in glosses if g.text]

        if not word or not reading or not meanings:
            elem.clear()
            continue

        # meaning — строка (первые 3 через запятую) для совместимости с пайплайном
        # meanings — полный список если понадобится позже
        words.append({
            "word": word,
            "reading": reading,
            "meaning": ", ".join(meanings[:3]),
            "meanings": meanings[:3],
            "source": "jmdict",
        })

        elem.clear()

    return words

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    words = parse_jmdict()

    with OUT_FILE.open("w", encoding="utf-8") as file:
        for word in words:
            file.write(json.dumps(word, ensure_ascii=False) + "\n")

    print(f"Saved {len(words)} words to {OUT_FILE}")

if __name__ == "__main__":
    main()