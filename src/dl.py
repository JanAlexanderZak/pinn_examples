class DeepLearningArguments:
    """ This class contains all deep learning arguments.
    """
    def __init__(
        self,
        seed: float,
        batch_size: int,
        min_epochs: int,
        max_epochs: int,
        sync_batchnorm: bool = True,
        num_workers: int = 1,
        accelerator: str = "cpu",
        devices: int = 1,
        sample_size: float = 1,
        pin_memory: bool = False,
        persistent_workers: bool = True,
    ) -> None:
        self.seed = float(seed)
        self.batch_size = int(batch_size)
        self.min_epochs = int(min_epochs)
        self.max_epochs = int(max_epochs)
        self.num_workers = int(num_workers)
        self.accelerator = accelerator
        self.sync_batchnorm = sync_batchnorm
        self.devices = int(devices)
        self.sample_size = float(sample_size)
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
