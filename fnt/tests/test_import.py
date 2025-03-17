# test_import.py
try:
    from field_neuro.video.concat_videos import concat_videos
    print("Import successful!")
except ModuleNotFoundError as e:
    print(f"Import failed: {e}")
