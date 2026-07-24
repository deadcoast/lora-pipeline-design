"""One shared inventory of the flat image/caption files consumed by training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_DATASET_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)
UNSUPPORTED_DATASET_IMAGE_EXTENSIONS = frozenset(
    {".avif", ".gif", ".heic", ".heif", ".jfif", ".svg"}
)


@dataclass(frozen=True, slots=True)
class DatasetPairInventory:
    """Exact files participating in, or preventing, trainer-ready pairs."""

    folder: Path
    pairs: tuple[tuple[Path, Path], ...]
    image_files: tuple[Path, ...]
    caption_files: tuple[Path, ...]
    duplicate_stem_groups: tuple[tuple[Path, ...], ...]
    images_without_captions: tuple[Path, ...]
    captions_without_images: tuple[Path, ...]
    unsupported_image_files: tuple[Path, ...]

    @property
    def structure_ready(self) -> bool:
        return (
            bool(self.pairs)
            and not self.duplicate_stem_groups
            and not self.images_without_captions
            and not self.captions_without_images
            and not self.unsupported_image_files
        )

    @property
    def review_pairs(self) -> tuple[tuple[Path, Path], ...]:
        """Pairs plus uniquely named images whose missing caption can be created."""

        duplicate_images = {
            path
            for group in self.duplicate_stem_groups
            for path in group
            if path.suffix.lower() in SUPPORTED_DATASET_IMAGE_EXTENSIONS
        }
        pair_by_image = {image: caption for image, caption in self.pairs}
        return tuple(
            (
                image,
                pair_by_image.get(image, image.with_suffix(".txt")),
            )
            for image in self.image_files
            if image not in duplicate_images
        )

    def problem_messages(self) -> tuple[str, ...]:
        """Plain-language, filename-specific structure problems."""

        messages: list[str] = []
        for group in self.duplicate_stem_groups:
            names = ", ".join(path.name for path in group)
            images = all(
                path.suffix.lower() in SUPPORTED_DATASET_IMAGE_EXTENSIONS
                for path in group
            )
            kind = "images" if images else "captions"
            messages.append(
                f"Multiple {kind} share one filename stem: {names}. "
                f"Keep one {kind[:-1]}."
            )
        for image in self.images_without_captions:
            messages.append(
                f"{image.name} has no caption. Create {image.with_suffix('.txt').name}."
            )
        for caption in self.captions_without_images:
            messages.append(
                f"{caption.name} has no supported image with the same filename stem."
            )
        for image in self.unsupported_image_files:
            messages.append(
                f"{image.name} is not a supported training image. Convert or remove it."
            )
        return tuple(messages)


def scan_dataset_pairs(folder: Path) -> DatasetPairInventory:
    """Inventory one flat dataset using the trainer's exact supported file set."""

    image_groups: dict[str, list[Path]] = {}
    caption_groups: dict[str, list[Path]] = {}
    unsupported: list[Path] = []
    if folder.is_dir():
        for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            stem = path.stem.casefold()
            if suffix in SUPPORTED_DATASET_IMAGE_EXTENSIONS:
                image_groups.setdefault(stem, []).append(path)
            elif suffix == ".txt":
                caption_groups.setdefault(stem, []).append(path)
            elif suffix in UNSUPPORTED_DATASET_IMAGE_EXTENSIONS:
                unsupported.append(path)

    duplicate_keys = {
        stem
        for stem in image_groups.keys() | caption_groups.keys()
        if len(image_groups.get(stem, ())) > 1 or len(caption_groups.get(stem, ())) > 1
    }
    duplicate_groups = tuple(
        group
        for stem in sorted(duplicate_keys)
        for group in (
            (
                tuple(sorted(image_groups[stem]))
                if len(image_groups.get(stem, ())) > 1
                else ()
            ),
            (
                tuple(sorted(caption_groups[stem]))
                if len(caption_groups.get(stem, ())) > 1
                else ()
            ),
        )
        if group
    )
    pair_keys = sorted((image_groups.keys() & caption_groups.keys()) - duplicate_keys)
    pairs = tuple(
        (image_groups[stem][0], caption_groups[stem][0]) for stem in pair_keys
    )
    images_without_captions = tuple(
        path
        for stem in sorted(image_groups.keys() - caption_groups.keys())
        for path in image_groups[stem]
    )
    captions_without_images = tuple(
        path
        for stem in sorted(caption_groups.keys() - image_groups.keys())
        for path in caption_groups[stem]
    )
    return DatasetPairInventory(
        folder=folder,
        pairs=pairs,
        image_files=tuple(
            path for stem in sorted(image_groups) for path in image_groups[stem]
        ),
        caption_files=tuple(
            path for stem in sorted(caption_groups) for path in caption_groups[stem]
        ),
        duplicate_stem_groups=duplicate_groups,
        images_without_captions=images_without_captions,
        captions_without_images=captions_without_images,
        unsupported_image_files=tuple(unsupported),
    )
