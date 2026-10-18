"""Generate FLASH Stage I semantic IDs from precomputed item embeddings."""

if __package__:
    from .flash_sid.cli import main
else:
    from flash_sid.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
