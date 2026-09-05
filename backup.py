import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKUP_DIR = BASE_DIR / "backups"

def create_backup(tag: str = "stable_v1"):
    BACKUP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"presales_stack_{tag}_{timestamp}.zip"
    archive_path = BACKUP_DIR / archive_name

    # Files to include in full version snapshot
    files_to_backup = [
        "app.py",
        "task_board.py",
        "agent_server.py",
        "start_stack.py",
        "README.md",
        ".env",
        "crm.db",
        "tasks.db",
        ".gitignore",
    ]

    print("=" * 65)
    print(f"Creating snapshot archive: {archive_name}")
    print("=" * 65)

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fname in files_to_backup:
            fpath = BASE_DIR / fname
            if fpath.exists():
                zipf.write(fpath, arcname=fname)
                size_kb = round(fpath.stat().st_size / 1024.0, 1)
                print(f"  [+] Added: {fname:<20} ({size_kb} KB)")
            else:
                print(f"  [-] Skipped (not found): {fname}")

    archive_size_kb = round(archive_path.stat().st_size / 1024.0, 1)
    print("=" * 65)
    print(f" SUCCESS: Backup saved to:")
    print(f"   {archive_path} ({archive_size_kb} KB)")
    print("=" * 65)
    return archive_path

if __name__ == "__main__":
    import sys
    custom_tag = sys.argv[1] if len(sys.argv) > 1 else "stable_v1"
    create_backup(custom_tag)
