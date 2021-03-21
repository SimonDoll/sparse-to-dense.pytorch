from dataloaders.carla_dataloader import CarlaDataset

"""
Small utility for debugging the carla data loader
"""

if __name__ == "__main__":
    dataset = CarlaDataset("data/carla", "train")
    dataset[40]
