import re

from defusedxml import ElementTree as ET

from exceptions import OptimizationError

# Event handler attributes to strip
EVENT_HANDLERS = {
    "onload", "onerror", "onclick", "onmouseover", "onmouseout",
    "onmousedown", "onmouseup", "onmousemove", "onfocus", "onblur",
    "onchange", "onsubmit", "onreset", "onselect", "onkeydown",
    "onkeypress", "onkeyup", "onabort", "onactivate", "onbegin",
    "onend", "onrepeat", "onunload", "onscroll", "onresize",
    "oninput", "onanimationstart", "onanimationend", "onanimationiteration",
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

    Args:
        data: Raw SVG bytes (UTF-8 encoded XML).

    Returns:
        Sanitized SVG bytes.

    Raises:
        OptimizationError: If SVG is malformed XML.
    """
    try:
        root = ET.fromstring(data)
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
