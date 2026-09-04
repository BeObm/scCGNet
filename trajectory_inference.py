import os
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx


def trajectory_inference(
    z,
    predicted_labels,
    output_dir,
    root_cluster=None,
    n_neighbors=15,
    random_state=111
):
    """
    Perform trajectory inference from a learned latent representation.

    Parameters
    ----------
    z : array-like or torch.Tensor
        Latent representation, shape (n_cells, latent_dim).

    predicted_labels : array-like or torch.Tensor
        Predicted cluster labels, shape (n_cells,).

    output_dir : str
        Directory where results will be saved.

    root_cluster : str/int or None
        Cluster used as root for DPT pseudotime.
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
    # Create output directory
    # --------------------------------------------------

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Convert PyTorch tensors to NumPy
    # --------------------------------------------------

    if hasattr(z, "detach"):
        z = z.detach().cpu().numpy()

    if hasattr(predicted_labels, "detach"):
        predicted_labels = predicted_labels.detach().cpu().numpy()

    z = np.asarray(z)
    predicted_labels = np.asarray(predicted_labels).reshape(-1)

    # --------------------------------------------------
    # Check input
    # --------------------------------------------------

    if z.ndim != 2:
        raise ValueError(
            f"z must be 2-dimensional. Got shape {z.shape}"
        )

    if z.shape[0] != len(predicted_labels):
        raise ValueError(
            f"Number of cells in z ({z.shape[0]}) does not "
            f"match number of labels ({len(predicted_labels)})"
        )

    print(f"Cells: {z.shape[0]}")
    print(f"Latent dimension: {z.shape[1]}")
    print(f"Clusters: {len(np.unique(predicted_labels))}")

    # --------------------------------------------------
    # Create AnnData
    # --------------------------------------------------

    adata = ad.AnnData(
        X=np.zeros((z.shape[0], 1), dtype=np.float32)
    )

    adata.obsm["X_scCGNet"] = z.astype(np.float32)

    adata.obs["cluster"] = predicted_labels.astype(str)

    # Make cluster categorical
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")

    # --------------------------------------------------
    # 1. Cell-cell neighbor graph
    # --------------------------------------------------

    print("Computing cell-cell graph...")

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

    # Plot UMAP
    fig, ax = plt.subplots(figsize=(8, 7))

    sc.pl.umap(
        adata,
        color="cluster",
        ax=ax,
        show=False,
        frameon=False
    )

    ax.set_title("scCGNet predicted clusters")

    fig.tight_layout()

    fig.savefig(
        os.path.join(output_dir, "umap_clusters.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------
    # 3. PAGA
    # --------------------------------------------------

    print("Computing PAGA...")

    sc.tl.paga(
        adata,
        groups="cluster"
    )

    # --------------------------------------------------
    # 4. Plot PAGA manually
    # --------------------------------------------------

    print("Plotting PAGA topology...")

    connectivities = adata.uns["paga"]["connectivities"]

    # Convert sparse matrix to NumPy
    if hasattr(connectivities, "toarray"):
        connectivities = connectivities.toarray()

    connectivities = np.asarray(connectivities)

    clusters = list(
        adata.obs["cluster"].cat.categories
    )

    # Create graph
    G = nx.Graph()

    for cluster in clusters:
        G.add_node(cluster)

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):

            weight = connectivities[i, j]

            if weight > 0:
                G.add_edge(
                    clusters[i],
                    clusters[j],
                    weight=float(weight)
                )

    # Layout
    pos = nx.spring_layout(
        G,
        seed=random_state,
        weight="weight"
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    # Nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=1000,
        ax=ax
    )

    # Labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        ax=ax
    )

    # Edges
    edge_widths = [
        1 + 5 * data["weight"]
        for _, _, data in G.edges(data=True)
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.7,
        ax=ax
    )

    ax.set_title("scCGNet PAGA trajectory topology")
    ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        os.path.join(output_dir, "paga.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------
    # 5. DPT pseudotime
    # --------------------------------------------------

    if root_cluster is not None:

        root_cluster = str(root_cluster)

        root_cells = np.where(
            adata.obs["cluster"].values == root_cluster
        )[0]

        if len(root_cells) == 0:

            raise ValueError(
                f"Root cluster '{root_cluster}' not found. "
                f"Available clusters: "
                f"{clusters}"
            )

        # Find cell closest to the center of root cluster
        root_z = z[root_cells].mean(axis=0)

        distances = np.linalg.norm(
            z[root_cells] - root_z,
            axis=1
        )

        root_cell = root_cells[np.argmin(distances)]

        print(f"Root cluster: {root_cluster}")
        print(f"Root cell: {root_cell}")

        adata.uns["iroot"] = int(root_cell)

        # Diffusion map
        print("Computing diffusion map...")

        sc.tl.diffmap(adata)

        # DPT
        print("Computing DPT pseudotime...")

        sc.tl.dpt(adata)

        # --------------------------------------------------
        # Save pseudotime
        # --------------------------------------------------

        pseudotime = np.asarray(
            adata.obs["dpt_pseudotime"]
        )

        np.savetxt(
            os.path.join(output_dir, "pseudotime.csv"),
            pseudotime,
            delimiter=","
        )

        # --------------------------------------------------
        # Plot pseudotime
        # --------------------------------------------------

        fig, ax = plt.subplots(figsize=(8, 7))

        sc.pl.umap(
            adata,
            color="dpt_pseudotime",
            ax=ax,
            show=False,
            frameon=False
        )

        ax.set_title("scCGNet pseudotime")

        fig.tight_layout()

        fig.savefig(
            os.path.join(output_dir, "umap_pseudotime.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    # --------------------------------------------------
    # Save AnnData
    # --------------------------------------------------

    output_file = os.path.join(
        output_dir,
        "trajectory.h5ad"
    )

    adata.write(output_file)

    print("\nTrajectory inference completed.")
    print(f"Results saved to: {output_dir}")

    return adata