"""
Helper script to download and extract prebuilt llama.cpp binaries.
This allows workshop participants to run fast C++ local inference 
without needing a C++ compiler or CMake build environment.
"""

import os
import sys
import zipfile
import urllib.request
import platform

# Official llama.cpp release repo
RELEASE_REPO = "ggml-org/llama.cpp"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "llama.cpp_bin")

def get_latest_release_url():
    """Fetch direct download URL for prebuilt binaries zip."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        # Direct download for Windows x64 CPU prebuilt release
        # ggml-org/llama.cpp releases tag format e.g. b4850
        tag_url = f"https://api.github.com/repos/{RELEASE_REPO}/releases/latest"
        try:
            req = urllib.request.Request(tag_url, headers={'User-Agent': 'Mozilla/5.0'})
            import json
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                for asset in data.get('assets', []):
                    name = asset.get('name', '')
                    if ('win-x64' in name or 'win-cpu-x64' in name) and name.endswith('.zip') and 'cuda' not in name and 'cudart' not in name:
                        return asset.get('browser_download_url'), name
        except Exception as e:
            print(f"Note: Could not query GitHub API automatically ({e}).")
    
    return None, None

def download_and_extract(url=None, zip_path=None):
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    if zip_path and os.path.exists(zip_path):
        print(f"Extracting local zip archive: {zip_path}...")
        archive = zip_path
    elif url:
        archive = os.path.join(TARGET_DIR, "llama_release.zip")
        print(f"Downloading prebuilt llama.cpp from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(archive, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete!")
    else:
        print("No zip or download URL available. Please download prebuilt release zip manually from:")
        print("https://github.com/ggml-org/llama.cpp/releases")
        return

    print(f"Extracting executables to {TARGET_DIR}...")
    with zipfile.ZipFile(archive, 'r') as zip_ref:
        zip_ref.extractall(TARGET_DIR)
    
    print(f"llama.cpp binaries setup successfully in: {TARGET_DIR}")

if __name__ == "__main__":
    print("=== llama.cpp Prebuilt Binary Setup ===")
    url, filename = get_latest_release_url()
    if url:
        print(f"Found release asset: {filename}")
        download_and_extract(url=url)
    else:
        # Fallback to local zip if user placed one in repo
        local_zips = [f for f in os.listdir('.') if f.endswith('.zip') and 'llama' in f.lower()]
        if local_zips:
            download_and_extract(zip_path=local_zips[0])
        else:
            download_and_extract()
