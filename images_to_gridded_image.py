from default_to import default_to
import math
import tensorflow as tf
from typing import Optional


def images_to_gridded_image(images: tf.Tensor) -> tf.Tensor:
    '''
        images should have shape [grid_height, grid_width, image_height, image_width, channel_count]
    '''
    grid_height, grid_width, image_height, image_width, channel_count = images.shape
    grid = tf.transpose(images, perm=[0, 2, 1, 3, 4])
    grid = tf.reshape(grid, (grid_height * image_height, grid_width * image_width, channel_count))
    return grid


def spherically_project_images_to_grid(
        images: tf.Tensor,
        background_color: Optional[tf.Tensor] = None) -> tf.Tensor:
    '''
        images should have shape [lat, lon, height, width, channels]
    '''
    grid_height, grid_width, _, _, _ = images.shape

    background_color = default_to(background_color, tf.constant([128, 128, 128], dtype=tf.uint8))

    assert background_color.dtype == images.dtype
    projected = tf.Variable(tf.broadcast_to(background_color, images.shape))
    cur_dist = tf.Variable(tf.fill((grid_height, grid_width), math.inf))

    for cell_y, unproj_cell_x in [(y, x) for y in range(grid_height) for x in range(grid_width)]:
        x_range = math.sin(cell_y / (grid_height - 1) * math.pi)
        proj_cell_x = unproj_cell_x * x_range + grid_width * 0.5 * (1 - x_range)
        int_proj_cell_x = round(proj_cell_x)
        cell_center_dist = tf.abs(int_proj_cell_x - proj_cell_x)
        if cell_center_dist < cur_dist[cell_y, int_proj_cell_x]:
            cur_dist[cell_y, int_proj_cell_x].assign(cell_center_dist)
            projected[cell_y, int_proj_cell_x].assign(images[cell_y, unproj_cell_x])

    return images_to_gridded_image(projected)
