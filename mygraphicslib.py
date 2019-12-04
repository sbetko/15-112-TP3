from helpers112 import rgbString

# Returns the color at the relative distance towards rgb2 from rgb1
# Takes RGB in string format and in 0-255 integer encoding
def mapPercentToLegendColor(percent, rgb1, rgb2):
        r1, g1, b1 = int(rgb1[:3]), int(rgb1[3:6]), int(rgb1[6:9])
        r2, g2, b2 = int(rgb2[:3]), int(rgb2[3:6]), int(rgb2[6:9])

        r3 = int((1 - percent)*r1 + percent*r2)
        g3 = int((1 - percent)*g1 + percent*g2)
        b3 = int((1 - percent)*b1 + percent*b2)

        return rgbString(r3, g3, b3)

def drawColorGradientVertical(canvas, x, y, width, height, rgb1, rgb2):
    canvas.create_rectangle(x, y, x + width, y + height)
    numPixels = int(height)
    for px in range(numPixels):
        percent = px / numPixels
        canvas.create_line(x, y + px, x + width, y + px,
                           fill = mapPercentToLegendColor(percent,
                                                          rgb1,
                                                          rgb2))
    