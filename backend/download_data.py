import io
import os
import urllib.request
import zipfile

URL = "https://github.com/jukedeck/nottingham-dataset/archive/refs/heads/master.zip"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "midi")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Downloading dataset... (a few MB, please wait)")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read()

    count = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.lower().endswith((".mid", ".midi")):
                filename = os.path.basename(name)
                with open(os.path.join(OUT_DIR, filename), "wb") as f:
                    f.write(z.read(name))
                count += 1

    print(f"Done! Saved {count} MIDI files to {OUT_DIR}")


if __name__ == "__main__":
    main()