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

    tree = ET.parse(RAW_FILE)
    root = tree.getroot()

    for entry in root.findall("entry"):
    
        keb = entry.find("k_ele/keb")

        reb = entry.find("r_ele/reb")

        glosses = entry.findall("sense/gloss")

        if reb is None or not glosses:
            continue

        word = keb.text if keb is not None else reb.text
        reading = reb.text
        meanings = [gloss.text for gloss in glosses if gloss.text]

        if not word or not reading or not meanings:
            continue

        words.append({
            "word": word,
            "reading": reading,
            "meanings": meanings[:3],
            "source": "jmdict"
        })

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