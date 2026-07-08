import argparse

from src.config import IDENTITY_FILE, SELECTED_SUBSETS_DIR
from src.celeba_io import create_identity_pair_subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-identities", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_file = (
        SELECTED_SUBSETS_DIR
        / f"celeba_subset_{args.n_identities}_ids_seed_{args.seed}.json"
    )

    subset = create_identity_pair_subset(
        identity_file=IDENTITY_FILE,
        output_file=output_file,
        n_identities=args.n_identities,
        seed=args.seed,
    )

    print(f"Saved subset to: {output_file}")
    print(f"Selected identities: {len(subset['pairs'])}")


if __name__ == "__main__":
    main()
