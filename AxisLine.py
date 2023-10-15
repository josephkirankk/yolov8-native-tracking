import cv2

class AxisLine:
    def __init__(self, x1, y1, x2, y2, color, thickness):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color
        self.highlight = (0, 255, 255)
        self.thickness = thickness

    def draw(self, image):
        cv2.line(image, (self.x1, self.y1), (self.x2, self.y2), self.color, self.thickness)

    def __str__(self):
        return f"AxisLine(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, color={self.color}, thickness={self.thickness})"

    def draw_text(self, image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=1):
        cv2.putText(image, text, position, font, font_scale, color, thickness)

    def drawHighlight(self,state, image):
        if(state):
            cv2.line(image, (self.x1, self.y1), (self.x2, self.y2), self.color, self.thickness)
        else:
            cv2.line(image, (self.x1, self.y1), (self.x2, self.y2), self.highlight, self.thickness)

    @staticmethod
    def draw_axis_lines(frame, axis_lines):
        for line in axis_lines:
            line.draw(frame)