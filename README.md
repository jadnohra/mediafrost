# mediafrost

An image and video backup tool

# Usage Examples

Dry run, list duplicates
`python3 -m mediafrost --no-videos --log-debug --out-dir ./backup --zip  --naive-dups --dry-run --list-duplicates /media/user/DISK/some_photos/`

Backup images only
`python3 -m mediafrost --no-videos --log-debug --out-dir ./backup --zip  --naive-dups /media/user/DISK/some_photos/`

Backup video only, encode (large) .MTS files
`python3 -m mediafrost --no-images --log-debug --out-dir ./backup --mts-encode-sh mediafrost/encode_video.sh --zip  --naive-dups /media/user/DISK/some_photos/`

# Help

```
python3 -m mediafrost --help
usage: __main__.py [-h] [--out-dir OUT_DIR] [--zip] [--struct-only]
                   [--no-hints] [--list-dups] [--naive-dups] [--no-images]
                   [--no-videos] [--mts-encode-sh MTS_ENCODE_SH]
                   [--list-encoded] [--log-debug] [--dry-run]
                   [--dry-run-no-tree] [--dry-run-detailed]
                   directory

positional arguments:
  directory

optional arguments:
  -h, --help            show this help message and exit
  --out-dir OUT_DIR
  --zip
  --struct-only
  --no-hints
  --list-dups
  --naive-dups
  --no-images
  --no-videos
  --mts-encode-sh MTS_ENCODE_SH
  --list-encoded
  --log-debug
  --dry-run
  --dry-run-no-tree
  --dry-run-detailed
```
