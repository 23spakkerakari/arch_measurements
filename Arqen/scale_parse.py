import re


'''
Helper file to convert arch scale string to px_per_unit
'''

def _parse_fraction_or_float(s: str) -> float:

    # "1/16" -> 0.0625
    s = s.strip()
    if "/" in s:
        num, den = s.split("/")
        return float(num) / float(den)
    return float(s)


def _parse_arch_length_to_inches(s: str) -> float:
    """
    Parse a paper/drawing-side length into inches.
    Supports examples:
      '1/16"'
      '1in'
      '0.25in'
      '10mm'
      '2.5cm'
    """
    s = s.strip().lower().replace(" ", "")

    if s.endswith('"'):
        return _parse_fraction_or_float(s[:-1])

    if s.endswith("in"):
        return _parse_fraction_or_float(s[:-2])

    if s.endswith("mm"):
        mm = float(s[:-2])
        return mm / 25.4

    if s.endswith("cm"):
        cm = float(s[:-2])
        return cm / 2.54

    raise ValueError(f"Unsupported drawing length: {s}")


def _parse_real_length_to_feet(s: str) -> float:
    """
    Parse a real-world length into feet.
    Supports examples:
      "1'-0\""
      "32'"
      "1ft"
      "12in"
      "3m"
      "2500mm"
    """
    s = s.strip().lower().replace(" ", "")

    # Architectural feet-inches form, e.g. 1'-0", 12'-6"
    m = re.fullmatch(r"(\d+)'-(\d+(?:\.\d+)?)\"?", s)
    if m:
        feet = float(m.group(1))
        inches = float(m.group(2))
        return feet + inches / 12.0

    # Feet with apostrophe, e.g. 32'
    m = re.fullmatch(r"(\d+(?:\.\d+)?)'", s)
    if m:
        return float(m.group(1))

    if s.endswith("ft"):
        return float(s[:-2])

    if s.endswith("feet"):
        return float(s[:-4])

    if s.endswith("foot"):
        return float(s[:-4])

    if s.endswith("in"):
        inches = float(s[:-2])
        return inches / 12.0

    if s.endswith('"'):
        inches = float(s[:-1])
        return inches / 12.0

    # if s.endswith("m"): #revisit this; unsure
    #     meters = float(s[:-1])
    #     return meters * 3.28084

    if s.endswith("mm"):
        mm = float(s[:-2])
        return (mm / 1000.0) * 3.28084

    raise ValueError(f"Unsupported real-world length: {s}")


def parse_scale(scale_str: str, dpi: int, output_unit: str = "ft") -> dict:
    """
    Return calibration info:
      {
        "px_per_unit": ...,
        "unit_label": "ft" or "m"
      }

    Supported:
      1/16" = 1'-0"
      1/4in = 1ft
      1" = 32'
      1:100
    """
    s = scale_str.strip().lower()

    if "=" in s:
        left, right = [part.strip() for part in s.split("=", 1)]

        drawing_inches = _parse_arch_length_to_inches(left)
        real_feet = _parse_real_length_to_feet(right)

        px_per_drawing_inch = dpi #kept at 300 for now, assumed to be high resolution
        px_per_foot = (drawing_inches * px_per_drawing_inch) / real_feet

        if output_unit == "ft":
            return {"px_per_unit": px_per_foot, "unit_label": "ft"}

        if output_unit == "m":
            px_per_meter = px_per_foot * 3.28084
            return {"px_per_unit": px_per_meter, "unit_label": "m"}

        raise ValueError(f"Unsupported output unit: {output_unit}")

    if ":" in s: #"1:100" -> 100
        left, right = [part.strip() for part in s.split(":", 1)]
        left_val = float(left)
        right_val = float(right)

        ratio = right_val / left_val

        # Assume drawing unit is inches because raster DPI is in inches.
        # So 1 drawing inch on paper corresponds to `ratio` real inches.
        px_per_drawing_inch = dpi
        real_inches_per_drawing_inch = ratio

        if output_unit == "ft":
            real_feet_per_drawing_inch = real_inches_per_drawing_inch / 12.0
            px_per_foot = px_per_drawing_inch / real_feet_per_drawing_inch
            return {"px_per_unit": px_per_foot, "unit_label": "ft"}

        if output_unit == "m":
            real_meters_per_drawing_inch = real_inches_per_drawing_inch * 0.0254
            px_per_meter = px_per_drawing_inch / real_meters_per_drawing_inch
            return {"px_per_unit": px_per_meter, "unit_label": "m"}

        raise ValueError(f"Unsupported output unit: {output_unit}")

    raise ValueError(f"Cannot parse scale: {scale_str}")