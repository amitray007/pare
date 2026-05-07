import re

from defusedxml import ElementTree as ET

from exceptions import OptimizationError

# Event handler attributes to strip
EVENT_HANDLERS = {
    "onload",
    "onerror",
    "onclick",
    "onmouseover",
    "onmouseout",
    "onmousedown",
    "onmouseup",
    "onmousemove",
    "onfocus",
    "onblur",
    "onchange",
    "onsubmit",
    "onreset",
    "onselect",
    "onkeydown",
    "onkeypress",
    "onkeyup",
    "onabort",
    "onactivate",
    "onbegin",
    "onend",
    "onrepeat",
    "onunload",
    "onscroll",
    "onresize",
    "oninput",
    "onanimationstart",
    "onanimationend",
    "onanimationiteration",
    "ontransitionend",
}

# Elements to remove entirely
DANGEROUS_ELEMENTS = {
    "script",
    "foreignobject",
}


def sanitize_svg(data: bytes) -> bytes:
    """Sanitize SVG content to remove security threats.

    Uses defusedxml to prevent XXE attacks during parsing.
    Strips all script tags, event handlers, external references,
    and data: URIs.

    Internal entity declarations (e.g. Wikimedia SVGs that use <!ENTITY st0 "…">
    as CSS shorthand) are allowed — they are pure string substitution within the
    document and involve no external fetching.  XXE is still blocked via
    forbid_external=True, which rejects any SYSTEM/PUBLIC entity that tries to
    resolve an external resource.  Recursive entity expansion (billion-laughs) is
    bounded by expat's built-in amplification limit (~8192× by default).

    Args:
        data: Raw SVG bytes (UTF-8 encoded XML).

    Returns:
        Sanitized SVG bytes.

    Raises:
        OptimizationError: If SVG is malformed XML.
    """
    try:
        # forbid_entities=False: allow safe internal entity declarations.
        # forbid_external=True: still block SYSTEM/PUBLIC external entity refs (XXE).
        # forbid_dtd left at False default so the DOCTYPE block that houses the
        # internal declarations is permitted.
        root = ET.fromstring(data, forbid_entities=False, forbid_external=True)
    except ET.ParseError as e:
        raise OptimizationError(f"Malformed SVG XML: {e}")

    _strip_dangerous_elements(root)
    _strip_event_handlers(root)
    _strip_dangerous_hrefs(root)
    _strip_css_imports(root)

    return ET.tostring(root, encoding="unicode").encode("utf-8")


def _strip_dangerous_elements(root):
    """Remove <script>, <foreignObject>, and similar elements."""
    # Collect elements to remove (can't modify tree during iteration)
    to_remove = []
    for element in root.iter():
        local_name = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        if local_name.lower() in DANGEROUS_ELEMENTS:
            to_remove.append(element)

    for element in to_remove:
        parent = _find_parent(root, element)
        if parent is not None:
            parent.remove(element)


def _strip_event_handlers(root):
    """Remove on* event handler attributes from all elements."""
    for element in root.iter():
        attrs_to_remove = []
        for attr in element.attrib:
            # Handle namespaced attributes
            local_attr = attr.split("}")[-1] if "}" in attr else attr
            if local_attr.lower() in EVENT_HANDLERS:
                attrs_to_remove.append(attr)
        for attr in attrs_to_remove:
            del element.attrib[attr]


def _strip_dangerous_hrefs(root):
    """Remove data:text/html URIs and external references in <use> href."""
    for element in root.iter():
        for attr_name in list(element.attrib):
            if "href" not in attr_name.lower():
                continue
            value = element.attrib[attr_name].strip()
            # Block data:text/html (XSS vector)
            if value.startswith("data:") and "text/html" in value.lower():
                del element.attrib[attr_name]
            # Block external URLs in <use> elements
            elif _is_external_url(value):
                local_tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
                if local_tag.lower() == "use":
                    del element.attrib[attr_name]


def _strip_css_imports(root):
    """Strip @import url() rules from <style> elements."""
    for element in root.iter():
        local_name = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        if local_name.lower() == "style" and element.text:
            element.text = re.sub(
                r"@import\s+url\s*\([^)]*\)\s*;?",
                "",
                element.text,
            )


def _find_parent(root, target):
    """Find the parent of a target element in the XML tree."""
    for parent in root.iter():
        for child in parent:
            if child is target:
                return parent
    return None


def _is_external_url(value: str) -> bool:
    """Check if a URL is external (http/https)."""
    return value.startswith("http://") or value.startswith("https://")
