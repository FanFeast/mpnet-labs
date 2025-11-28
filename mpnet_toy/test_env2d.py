from env2d import World2D

if __name__ == "__main__":
    world = World2D.random_world(num_obstacles=3, grid_size=64, seed=0)
    grid = world.get_occupancy_grid()
    print("Grid shape:", grid.shape)
    print("Fraction occupied:", grid.mean())

    p = world.sample_free_point()
    print("Sampled free point:", p)
    print("Point in collision:", world.is_point_in_collision(p[0], p[1]))

    # Test a segment: start in free, end in free
    p0 = world.sample_free_point()
    p1 = world.sample_free_point()
    collides = world.is_segment_in_collision(p0, p1)
    print("Segment collision:", collides)
