import os
import numpy as np
import scanpy as sc
import anndata as ad


def trajectory_inference(
    z,
    predicted_labels,
    output_dir="trajectory_results",
    root_cluster=None,
    n_neighbors=15,
    random_state=111
):
    """
    Trajectory inference from a learned latent representation.

    Parameters
    ----------
    z : array-like
        Latent representation, shape (n_cells, latent_dim).

    predicted_labels : array-like
        Predicted cluster labels, shape (n_cells,).

    output_dir : str
        Folder where all results will be saved.

    root_cluster : str/int or None
        Cluster used as root for pseudotime.
        If None, only PAGA topology is computed.

    n_neighbors : int
        Number of neighbors used to construct the cell graph.

    random_state : int
        Random seed.

    Returns
    -------
    adata : AnnData
    """

    # --------------------------------------------------
    # Create output folder
    # --------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Convert tensors to numpy
    # --------------------------------------------------
    if hasattr(z, "detach"):
        z = z.detach().cpu().numpy()

    if hasattr(predicted_labels, "detach"):
        predicted_labels = predicted_labels.detach().cpu().numpy()

    z = np.asarray(z)
    predicted_labels = np.asarray(predicted_labels)

    # --------------------------------------------------
    # Check input
    # --------------------------------------------------
    if z.ndim != 2:
        raise ValueError(
            f"z must have shape (n_cells, latent_dim), got {z.shape}"
        )

    if predicted_labels.ndim != 1:
        predicted_labels = predicted_labels.reshape(-1)

    if z.shape[0] != len(predicted_labels):
        raise ValueError(
            f"Number of cells in z ({z.shape[0]}) does not match "
            f"number of labels ({len(predicted_labels)})"
        )

    print(f"Number of cells: {z.shape[0]}")
    print(f"Latent dimension: {z.shape[1]}")
    print(f"Number of clusters: {len(np.unique(predicted_labels))}")

    # --------------------------------------------------
    # Create AnnData
    # --------------------------------------------------
    adata = ad.AnnData(
        X=np.zeros((z.shape[0], 1), dtype=np.float32)
    )

    adata.obsm["X_scCGNet"] = z.astype(np.float32)

    adata.obs["cluster"] = predicted_labels.astype(str)

    # --------------------------------------------------
    # 1. Construct cell-cell graph
    # --------------------------------------------------
    print("Computing nearest-neighbor graph...")

    sc.pp.neighbors(
        adata,
        use_rep="X_scCGNet",
        n_neighbors=n_neighbors,
        random_state=random_state
    )

    # --------------------------------------------------
    # 2. UMAP
    # --------------------------------------------------
    print("Computing UMAP...")

    sc.tl.umap(
        adata,
        random_state=random_state
    )

    # Save UMAP colored by predicted clusters
    sc.pl.umap(
        adata,
        color="cluster",
        title="scCGNet predicted clusters",
        show=False,
        save="_clusters.png"
    )

    # --------------------------------------------------
    # 3. PAGA
    # --------------------------------------------------
    print("Computing PAGA...")

    sc.tl.paga(
        adata,
        groups="cluster"
    )

    # Save PAGA graph
    sc.pl.paga(
        adata,
        title="scCGNet trajectory topology",
        show=False,
        save="_paga.png"
    )

    # --------------------------------------------------
    # 4. Diffusion map + DPT
    # --------------------------------------------------
    if root_cluster is not None:

        root_cluster = str(root_cluster)

        root_cells = np.where(
            adata.obs["cluster"].values == root_cluster
        )[0]

        if len(root_cells) == 0:
            raise ValueError(
                f"Root cluster '{root_cluster}' not found.\n"
                f"Available clusters: "
                f"{adata.obs['cluster'].unique().tolist()}"
            )

        # Find the cell closest to the center of the root cluster
        root_z = z[root_cells].mean(axis=0)

        distances = np.linalg.norm(
            z[root_cells] - root_z,
            axis=1
        )

        root_cell = root_cells[np.argmin(distances)]

        print(f"Root cluster: {root_cluster}")
        print(f"Root cell: {root_cell}")

        adata.uns["iroot"] = root_cell

        print("Computing diffusion map...")

        sc.tl.diffmap(adata)

        print("Computing DPT pseudotime...")

        sc.tl.dpt(adata)

        # Save pseudotime
        sc.pl.umap(
            adata,
            color="dpt_pseudotime",
            title="scCGNet pseudotime",
            show=False,
            save="_pseudotime.png"
        )

        # Save pseudotime values
        np.savetxt(
            os.path.join(output_dir, "pseudotime.csv"),
            adata.obs["dpt_pseudotime"].values,
            delimiter=","
        )

    else:
        print(
            "No root cluster provided. "
            "PAGA topology computed; pseudotime not calculated."
        )

    # --------------------------------------------------
    # Save AnnData
    # --------------------------------------------------
    output_file = os.path.join(
        output_dir,
        "trajectory.h5ad"
    )

    adata.write(output_file)

    print("\nTrajectory analysis completed.")
    print(f"Results saved in: {output_dir}")
    print(f"AnnData: {output_file}")

    return adata