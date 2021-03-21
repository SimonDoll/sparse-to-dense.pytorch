from dataloaders.carla_dataloader import CarlaDataset

if __name__ == "__main__":
    dataset = CarlaDataset("data/carla", "train")
    dataset[0]
