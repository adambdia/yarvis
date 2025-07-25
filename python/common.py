import numpy as np
import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


WINDOW_HEIGHT = 1080
WINDOW_WIDTH = 1920
WINDOW_NAME = 'yarvis'

IR_WIDTH = 512
IR_HEIGHT = 424

NUM_HANDS = 1
HAND_DETECTION_CONFIDENCE = 0.8

def cv_to_pygame(frame):
    tmp = np.copy(frame)
    tmp = np.rot90(tmp)
    tmp = pygame.surfarray.make_surface(tmp)
    return tmp

# draw text func
def draw_text(text, pos, colour, font, screen):
    surface = font.render(text, True, colour)
    screen.blit(surface, pos)