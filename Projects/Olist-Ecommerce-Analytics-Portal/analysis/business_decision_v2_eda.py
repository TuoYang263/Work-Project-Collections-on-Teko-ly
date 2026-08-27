from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery
from pandas.plotting import scatter_matrix


PROJECT_ID = "balmy-nuance-468118-g4"

OUTPUT_DIR = Path("analysis/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


QUERY = """
SELECT
  orders.order_id,

  IF(
    orders.avg_review_score <= 2,
    1,
    0
  ) AS negative_review,

  IF(
    orders.is_late_delivery,
    1,
    0
  ) AS is_late_delivery,

  orders.days_purchase_to_customer_delivery
    AS delivery_days,

  orders.order_gross_value,

  SAFE_DIVIDE(
    orders.freight_total_value,
    NULLIF(orders.order_gross_value, 0)
  ) AS freight_share,

  orders.order_total_items,

  IF(
    orders.order_total_items > 1,
    1,
    0
  ) AS multi_item_order

FROM
  `balmy-nuance-468118-g4.olist_marts.fct_orders`
  AS orders

WHERE
  orders.avg_review_score IS NOT NULL

  AND
  orders.order_delivered_customer_date
    IS NOT NULL

  AND
  orders.order_estimated_delivery_date
    IS NOT NULL

  AND
  orders.days_purchase_to_customer_delivery
    IS NOT NULL
"""


FEATURE_COLUMNS = [
    "negative_review",
    "is_late_delivery",
    "delivery_days",
    "order_gross_value",
    "freight_share",
    "order_total_items",
    "multi_item_order",
]


CONTINUOUS_COLUMNS = [
    "delivery_days",
    "order_gross_value",
    "freight_share",
    "order_total_items",
]


def save_correlation_heatmap(
    correlation: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    image = ax.imshow(
        correlation.values,
        vmin=-1,
        vmax=1,
    )

    labels = correlation.columns.tolist()

    ax.set_xticks(
        range(len(labels)),
        labels=labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(len(labels)),
        labels=labels,
    )

    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
            )

    ax.set_title(title)

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(fig)


def build_target_summary(
    data: pd.DataFrame,
    group_column: str,
    analysis_name: str,
) -> pd.DataFrame:
    result = (
        data.groupby(
            group_column,
            observed=True,
        )
        .agg(
            orders=(
                "negative_review",
                "size",
            ),
            negative_reviews=(
                "negative_review",
                "sum",
            ),
            negative_review_rate=(
                "negative_review",
                "mean",
            ),
        )
        .reset_index()
    )

    result.insert(
        0,
        "analysis",
        analysis_name,
    )

    result = result.rename(
        columns={
            group_column: "group",
        }
    )

    return result


def main() -> None:
    client = bigquery.Client(
        project=PROJECT_ID
    )

    print(
        "Loading eligible order-level data..."
    )

    data = (
        client.query(QUERY)
        .result()
        .to_dataframe()
    )

    data["order_gross_value"] = (
        data["order_gross_value"].astype(float)
    )

    data["freight_share"] = (
        data["freight_share"].astype(float)
    )

    print()
    print("=== DATASET ===")
    print(f"Rows: {len(data):,}")

    print()
    print("=== MISSING VALUES ===")
    print(
        data[FEATURE_COLUMNS]
        .isna()
        .sum()
    )

    # -------------------------------------------------
    # Correlation analysis
    # -------------------------------------------------

    pearson = data[
        FEATURE_COLUMNS
    ].corr(
        method="pearson"
    )

    spearman = data[
        FEATURE_COLUMNS
    ].corr(
        method="spearman"
    )

    print()
    print("=== PEARSON CORRELATION ===")
    print(
        pearson.round(3)
    )

    print()
    print("=== SPEARMAN CORRELATION ===")
    print(
        spearman.round(3)
    )

    pearson.to_csv(
        OUTPUT_DIR
        / "pearson_correlation.csv"
    )

    spearman.to_csv(
        OUTPUT_DIR
        / "spearman_correlation.csv"
    )

    save_correlation_heatmap(
        pearson,
        "Pearson correlation",
        OUTPUT_DIR
        / "pearson_correlation.png",
    )

    save_correlation_heatmap(
        spearman,
        "Spearman correlation",
        OUTPUT_DIR
        / "spearman_correlation.png",
    )

    # -------------------------------------------------
    # Scatter matrix
    # -------------------------------------------------

    continuous_data = (
        data[CONTINUOUS_COLUMNS]
        .dropna()
    )

    sample_size = min(
        4000,
        len(continuous_data),
    )

    scatter_sample = (
        continuous_data.sample(
            n=sample_size,
            random_state=42,
        )
    )

    scatter_matrix(
        scatter_sample,
        figsize=(10, 10),
        diagonal="hist",
        alpha=0.15,
    )

    plt.suptitle(
        "Feature pair relationships",
        y=1.02,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "feature_scatter_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------
    # Target association
    # -------------------------------------------------

    target_data = data.copy()

    target_data["delivery_status"] = (
        np.where(
            target_data[
                "is_late_delivery"
            ]
            == 1,
            "late",
            "on_time",
        )
    )

    target_data["delivery_bucket"] = (
        pd.cut(
            target_data[
                "delivery_days"
            ],
            bins=[
                -np.inf,
                7,
                14,
                21,
                30,
                np.inf,
            ],
            labels=[
                "0-7",
                "8-14",
                "15-21",
                "22-30",
                "31+",
            ],
        )
    )

    target_data["freight_bucket"] = (
        pd.qcut(
            target_data[
                "freight_share"
            ],
            q=5,
            duplicates="drop",
        )
    )

    target_data["gmv_bucket"] = (
        pd.qcut(
            target_data[
                "order_gross_value"
            ],
            q=5,
            duplicates="drop",
        )
    )

    target_data["item_group"] = (
        np.where(
            target_data[
                "multi_item_order"
            ]
            == 1,
            "multi_item",
            "single_item",
        )
    )

    summaries = [
        build_target_summary(
            target_data,
            "delivery_status",
            "late_delivery",
        ),
        build_target_summary(
            target_data,
            "delivery_bucket",
            "delivery_days",
        ),
        build_target_summary(
            target_data,
            "freight_bucket",
            "freight_share",
        ),
        build_target_summary(
            target_data,
            "gmv_bucket",
            "order_gmv",
        ),
        build_target_summary(
            target_data,
            "item_group",
            "item_count",
        ),
    ]

    target_summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    print()
    print(
        "=== TARGET ASSOCIATION ==="
    )

    print(
        target_summary.to_string(
            index=False
        )
    )

    target_summary.to_csv(
        OUTPUT_DIR
        / "target_association.csv",
        index=False,
    )

    print()
    print(
        f"EDA outputs written to "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()