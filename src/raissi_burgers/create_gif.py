from src.visualization import create_gif


if __name__ == "__main__":
    create_gif(
        frame_folder="src/raissi_burgers/plots/epochs/",
        output_path="src/raissi_burgers/raissi_burgers.gif",
        duration=400,
    )
