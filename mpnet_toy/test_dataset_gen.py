# test_dataset_gen.py
from dataset_gen import generate_dataset_many_worlds, save_dataset_npz


def main() -> None:
    dataset = generate_dataset_many_worlds(
        num_worlds=4000,
        num_paths_per_world=10,
        grid_size=64,
        seed=0,
        num_threads=24,
    )

    print("Keys:", dataset.keys())
    print("grids shape:", dataset["grids"].shape)
    print("x_cur shape:", dataset["x_cur"].shape)
    print("x_goal shape:", dataset["x_goal"].shape)
    print("x_next shape:", dataset["x_next"].shape)

    # Inspect first sample
    print("First grid min/max:", dataset["grids"][0].min(), dataset["grids"][0].max())
    print("First x_cur:", dataset["x_cur"][0])
    print("First x_goal:", dataset["x_goal"][0])
    print("First x_next:", dataset["x_next"][0])

    # Optionally save to disk
    save_dataset_npz("mpnet_toy_dataset.npz", dataset)
    print("Saved dataset to mpnet_toy_dataset.npz")


if __name__ == "__main__":
    main()
