import numpy as np
import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def cv_to_pygame(frame):
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    return frame

# draw text func
def draw_text(text, pos, colour, font, screen):
    surface = font.render(text, True, colour)
    screen.blit(surface, pos)