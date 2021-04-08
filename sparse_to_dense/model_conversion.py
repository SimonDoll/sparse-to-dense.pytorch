from numpy.core.numeric import full
import torch
import argparse
import pathlib

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Utility to extract the models state dict from a full checkpoint")

    parser.add_argument("checkpoint", type=str,
                        help="Sparse to dense training output full checkpoint path")

    parser.add_argument(
        "--out", type=str, help="converted output file", default="sparse_to_dense.converted")

    args = parser.parse_args()

    full_checkpoint = pathlib.Path(args.checkpoint)
    assert full_checkpoint.is_file(), "checkpoint file does not exist"

    # convert the full model to its state dict
    torch_checkpoint = torch.load(full_checkpoint)

    model = torch_checkpoint["model"]
    print("saving models state dict to {}".format(args.out))
    torch.save(model.state_dict(), args.out)
