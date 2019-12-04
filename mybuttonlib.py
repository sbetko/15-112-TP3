from mymathlib import *

class Button(object):
    def __init__(self, function, text):
        self.function = function
        self.text = text
    
    def activate(self):
        self.function()
    
    def assignBounds(self, bounds):
        self.bounds = bounds

class Panel(object):
    def __init__(self, x, y, width, height, anchor = ""):
        self.width = width
        self.height = height
        self.buttons = []
        self.backgroundColor = None
        self.buttonBounds = []
        self.anchor = anchor

        if anchor == "":
            self.x = x - self.width / 2
            self.y = y - self.height / 2
        else:
            if "e" in anchor: self.x = x - self.width
            elif "w" in anchor: self.x = x
            if "n" in anchor: self.y = y
            elif "s" in anchor: self.y = y - self.height
        
    
    def addButton(self, button):
        self.buttons.append(button)
        self.regenerateButtonBounds()
    
    def regenerateButtonBounds(self):
        self.buttonBounds = []
        dHeight = self.height/len(self.buttons)
        dWidth = self.width
        for i in range(len(self.buttons)):
            x1, x2 = self.x, self.x + self.width
            y1, y2 = self.y + i*dHeight, self.y + (i+1)*dHeight
            bounds = (x1,y1,x2,y2)
            self.buttonBounds.append(bounds)
    
    def drawPanelVertical(self, canvas):
        x, y, width, height = self.x, self.y, self.width, self.height
        canvas.create_rectangle(x, y, x + width, y + height,
                                fill = self.backgroundColor)
        if len(self.buttons) == 0: return

        for i in range(len(self.buttons)):
            buttonText = self.buttons[i].text
            x1, y1, x2, y2 = self.buttonBounds[i]
            canvas.create_rectangle(x1, y1, x2, y2)
            xMid, yMid = (x1 + x2) / 2, (y1 + y2)/2
            canvas.create_text(xMid, yMid, text = buttonText)
    
    def getBounds(self):
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def mousePressed(self, point):
        for i in range(len(self.buttonBounds)):
            buttonBounds = self.buttonBounds[i]
            if pointInBounds(point, buttonBounds):
                self.buttons[i].activate()
                return