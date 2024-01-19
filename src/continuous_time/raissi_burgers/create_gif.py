import glob

from PIL import Image


def create_gif(
    frame_folder: str,
    save_path: str,
    duration: int = 400,
):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    frame_one = frames[0]
    frame_one.save(
        save_path,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=0)
    

if __name__ == "__main__":
    create_gif(
        "src/continuous_time/raissi_burgers/plots/epochs/",
        "src/continuous_time/raissi_burgers/raissi_burgers.gif",
        duration=400,
    )
