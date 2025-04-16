import pygame
pygame.init()


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 50

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

clock = pygame.time.Clock()

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for x in range(0, SCREEN_WIDTH, GRID_SIZE):
         pygame.draw.line(screen, (0, 0, 0), (x, 0), (x, SCREEN_HEIGHT))

    for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(screen, (0, 0, 0), (0, y), (SCREEN_WIDTH, y))

    clock.tick(60)

    pygame.display.flip()

pygame.quit()

def draw_board(board):