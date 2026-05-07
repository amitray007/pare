"""Stub synthesizers for fetched entries.

Registers the `fetched_photo` and `fetched_vector` content_kinds so that the
synthesize() dispatcher raises a clear, actionable error instead of 'unknown
content_kind' when a fetched entry is accidentally routed through the synthesis
path.

For `fetched_vector` (SVG/SVGZ): the builder fetches the source bytes directly
and passes them to the encoder as raw bytes — no Image.open() involved.
"""

from __future__ import annotations

from bench.corpus.synthesis._common import register_kind


@register_kind("fetched_photo")
def _fetched_photo_stub(*, seed: int, width: int, height: int, **params) -> None:  # type: ignore[return]
    raise RuntimeError(
        "content_kind 'fetched_photo' is not synthesized — the builder must "
        "fetch this entry's bytes via SourceSpec instead. If you reached this "
        "via a manual synthesize() call, route through the build() pipeline."
    )


@register_kind("fetched_vector")
def _fetched_vector_stub(*, seed: int, width: int, height: int, **params) -> None:  # type: ignore[return]
    raise RuntimeError(
        "content_kind 'fetched_vector' is not synthesized — the builder must "
        "fetch this entry's bytes via SourceSpec and write them as-is (vector "
        "pass-through). If you reached this via a manual synthesize() call, "
        "route through the build() pipeline."
    )


@register_kind("fetched_text_screenshot")
def _fetched_text_screenshot_stub(*, seed: int, width: int, height: int, **params) -> None:  # type: ignore[return]
    raise RuntimeError(
        "content_kind 'fetched_text_screenshot' is not synthesized — the builder must "
        "fetch this entry's bytes via SourceSpec instead. Entries in this category "
        "are real-world UI, code editor, or dashboard screenshots."
    )


@register_kind("fetched_graphic_palette")
def _fetched_graphic_palette_stub(*, seed: int, width: int, height: int, **params) -> None:  # type: ignore[return]
    raise RuntimeError(
        "content_kind 'fetched_graphic_palette' is not synthesized — the builder must "
        "fetch this entry's bytes via SourceSpec instead. Entries in this category "
        "are real-world infographics, charts, or simple-palette illustrations."
    )


@register_kind("fetched_graphic_geometric")
def _fetched_graphic_geometric_stub(*, seed: int, width: int, height: int, **params) -> None:  # type: ignore[return]
    raise RuntimeError(
        "content_kind 'fetched_graphic_geometric' is not synthesized — the builder must "
        "fetch this entry's bytes via SourceSpec instead. Entries in this category "
        "are real-world technical diagrams, schematics, or line drawings."
    )


@register_kind("fetched_transparent_overlay")
def _fetched_transparent_overlay_stub(*, seed: int, width: int, height: int, **params) -> None:  # type: ignore[return]
    raise RuntimeError(
        "content_kind 'fetched_transparent_overlay' is not synthesized — the builder must "
        "fetch this entry's bytes via SourceSpec instead. Entries in this category "
        "are real-world PNG icons, logos with alpha, or badges."
    )


@register_kind("fetched_animated_redraw")
def _fetched_animated_redraw_stub(*, seed: int, width: int, height: int, **params) -> None:  # type: ignore[return]
    raise RuntimeError(
        "content_kind 'fetched_animated_redraw' is not synthesized — the builder must "
        "fetch this entry's bytes via SourceSpec instead. Entries in this category "
        "are real-world animated GIFs with full-frame redraws (loading spinners, "
        "animated charts, weather radar loops)."
    )


@register_kind("fetched_path_flat_text")
def _fetched_path_flat_text_stub(*, seed: int, width: int, height: int, **params) -> None:  # type: ignore[return]
    raise RuntimeError(
        "content_kind 'fetched_path_flat_text' is not synthesized — the builder must "
        "fetch this entry's bytes via SourceSpec instead. Entries in this category "
        "are real-world posters, comic panels, or infographics with flat colors and text."
    )
