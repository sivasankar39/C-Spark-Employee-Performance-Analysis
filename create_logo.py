from PIL import Image, ImageDraw, ImageFont
import os
import math


# =========================================================
# PATH
# =========================================================

output_dir = os.path.join(
    "flask_app",
    "static",
    "images"
)

os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "synq-logo.png"
)


# =========================================================
# CANVAS
# =========================================================

width = 900
height = 260

image = Image.new(
    "RGBA",
    (width, height),
    (0, 0, 0, 0)
)

draw = ImageDraw.Draw(image)


# =========================================================
# FONTS
# =========================================================

font_paths = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/calibrib.ttf"
]

font_path = None

for path in font_paths:
    if os.path.exists(path):
        font_path = path
        break


if font_path:

    logo_font = ImageFont.truetype(
        font_path,
        105
    )

    tagline_font = ImageFont.truetype(
        font_path,
        24
    )

else:

    logo_font = ImageFont.load_default()
    tagline_font = ImageFont.load_default()


# =========================================================
# COLORS
# =========================================================

purple = (168, 85, 247, 255)
light_purple = (216, 180, 254, 255)
white = (255, 255, 255, 255)
gray = (145, 145, 155, 255)


# =========================================================
# SYNQ SYMBOL
# =========================================================

cx = 115
cy = 112

radius = 72


# outer orbital ring

draw.arc(
    (
        cx - radius,
        cy - radius,
        cx + radius,
        cy + radius
    ),
    210,
    65,
    fill=purple,
    width=5
)


# second orbit

draw.arc(
    (
        cx - 58,
        cy - 58,
        cx + 58,
        cy + 58
    ),
    35,
    235,
    fill=light_purple,
    width=3
)


# =========================================================
# S SHAPE
# =========================================================

points = []

for i in range(101):

    t = i / 100

    x = cx - 38 + (76 * t)

    y = (
        cy
        - 40
        + 80 * t
        + 17 * math.sin(
            t * math.pi * 2
        )
    )

    points.append((x, y))


draw.line(
    points,
    fill=white,
    width=15,
    joint="curve"
)


# =========================================================
# NODE DOTS
# =========================================================

nodes = [
    (cx - 53, cy - 45),
    (cx + 50, cy - 25),
    (cx - 48, cy + 48),
    (cx + 55, cy + 42)
]

for x, y in nodes:

    draw.ellipse(
        (
            x - 5,
            y - 5,
            x + 5,
            y + 5
        ),
        fill=purple
    )


# =========================================================
# SYNQ TEXT
# =========================================================

text_x = 205
text_y = 55

draw.text(
    (text_x, text_y),
    "SYNQ",
    font=logo_font,
    fill=white,
    stroke_width=1,
    stroke_fill=white
)


# =========================================================
# TAGLINE
# =========================================================

draw.text(
    (text_x + 8, 172),
    "INTELLIGENT WORKFORCE ANALYTICS",
    font=tagline_font,
    fill=gray
)


# =========================================================
# SAVE
# =========================================================

image.save(
    output_file,
    "PNG"
)

print()
print("======================================")
print("SYNQ LOGO CREATED SUCCESSFULLY")
print("======================================")
print()
print("Location:")
print(output_file)
print()