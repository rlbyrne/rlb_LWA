import subprocess
from pathlib import Path

# --- Configuration ---
DIRECTORY_TO_SEARCH = "/lustre/pipeline/cosmology/82MHz/2026-04-22/05"  # Change this to your target directory path
TARGET_SIZE_MB = 339
TOLERANCE_MB = 1  # Allows size to be between 338M and 340M
DRY_RUN = False    # Set to False to actually delete the files
# ---------------------

def get_dir_size_mb(path: Path) -> int:
    """Uses the system 'du' command to quickly get directory size in MB."""
    try:
        # -s: summary (total only), -m: output in Megabytes
        result = subprocess.run(
            ['du', '-sm', str(path)], 
            capture_output=True, 
            text=True, 
            check=True
        )
        # Output format is "339\t/path/to/directory\n", so we split and take the first item
        size_mb_str = result.stdout.split()[0]
        return int(size_mb_str)
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Could not read size for {path}: {e}")
        return -1
    except FileNotFoundError:
        print("[ERROR] The 'du' command is not available on this system.")
        return -1

def main():
    target_dir = Path(DIRECTORY_TO_SEARCH)

    # Find all files ending in .ms.tar.gz
    for tarball in target_dir.rglob("*.ms.tar.gz"):
        
        # Remove the '.tar.gz' (7 characters) to get the expected directory name
        expected_dir_name = tarball.name[:-7] 
        ms_dir = tarball.parent / expected_dir_name

        if ms_dir.is_dir():
            dir_size_mb = get_dir_size_mb(ms_dir)
            
            # Skip if there was an error reading the directory size
            if dir_size_mb == -1:
                continue
            
            # Check if the directory size matches our target (within tolerance)
            if abs(dir_size_mb - TARGET_SIZE_MB) <= TOLERANCE_MB:
                print(f"[MATCH] Found tarball: {tarball.name}")
                print(f"        Matching dir : {ms_dir.name} (~{dir_size_mb} MB)")
                
                if DRY_RUN:
                    print(f"        [DRY RUN] Would delete: {tarball}")
                else:
                    print(f"        [ACTION] Deleting: {tarball}")
                    tarball.unlink() # Deletes the file
            else:
                pass 
                # Uncomment below to see directories that exist but failed the size check
                # print(f"[SKIP] Directory {ms_dir.name} exists but size is {dir_size_mb} MB")

if __name__ == "__main__":
    main()