from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from google.cloud import bigquery
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)
from datetime import (
    datetime,
    timezone,
)
from sklearn.model_selection import (
    StratifiedKFold,
)


PROJECT_ID = "balmy-nuance-468118-g4"

OUTPUT_DIR = Path("analysis/output")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_ID = (
    "balmy-nuance-468118-g4."
    "olist_analytics."
    "analytics_state_diagnostics_v2"
)

MODEL_VERSION = (
    "business_decision_v2_logit_001"
)


QUERY = """
SELECT
  customers.customer_state
    AS state_code,

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

  IF(
    orders.order_total_items > 1,
    1,
    0
  ) AS multi_item_order

FROM
  `balmy-nuance-468118-g4.olist_marts.fct_orders`
  AS orders

JOIN
  `balmy-nuance-468118-g4.olist_marts.dim_customers`
  AS customers

  ON customers.customer_id =
     orders.customer_id

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


FEATURES = [
    "is_late_delivery",
    "delivery_days_log",
    "gmv_log",
    "multi_item_order",
]


def classify_diagnostic(
    row: pd.Series,
) -> str:
    if row["orders"] < 100:
        return "INSUFFICIENT_EVIDENCE"

    if (
        row["residual_pp"] >= 1.0
        and row["ci_lower_pp"] > 0
    ):
        return "WORSE_THAN_EXPECTED"

    if (
        row["residual_pp"] <= -1.0
        and row["ci_upper_pp"] < 0
    ):
        return "BETTER_THAN_EXPECTED"

    return "AS_EXPECTED"


def main() -> None:
    client = bigquery.Client(
        project=PROJECT_ID
    )

    print(
        "Loading state diagnostic data..."
    )

    data = (
        client.query(QUERY)
        .result()
        .to_dataframe()
    )

    data["order_gross_value"] = (
        data["order_gross_value"]
        .astype(float)
    )

    data["delivery_days_log"] = (
        np.log1p(
            data["delivery_days"]
            .astype(float)
        )
    )

    data["gmv_log"] = (
        np.log1p(
            data["order_gross_value"]
        )
    )

    X = data[FEATURES].astype(float)

    y = (
        data["negative_review"]
        .astype(int)
    )

    # -----------------------------------------
    # Cross-fitted out-of-fold probabilities
    # -----------------------------------------

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    probabilities = np.empty(
        len(data),
        dtype=float,
    )

    print()
    print(
        "Generating 5-fold "
        "out-of-fold predictions..."
    )

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        folds.split(X, y),
        start=1,
    ):
        X_train = sm.add_constant(
            X.iloc[train_index],
            has_constant="add",
        )

        X_test = sm.add_constant(
            X.iloc[test_index],
            has_constant="add",
        )

        y_train = y.iloc[train_index]

        model = sm.Logit(
            y_train,
            X_train,
        )

        result = model.fit(
            disp=False
        )

        probabilities[test_index] = (
            result.predict(X_test)
        )

        print(
            f"Fold {fold_number}/5 complete"
        )

    data["expected_probability"] = (
        probabilities
    )

    # -----------------------------------------
    # Global out-of-fold sanity check
    # -----------------------------------------

    auc = roc_auc_score(
        y,
        probabilities,
    )

    brier = brier_score_loss(
        y,
        probabilities,
    )

    print()
    print(
        "=== OUT-OF-FOLD PERFORMANCE ==="
    )

    print(
        f"Actual negative-review rate: "
        f"{y.mean():.4f}"
    )

    print(
        f"Mean expected probability: "
        f"{probabilities.mean():.4f}"
    )

    print(
        f"ROC AUC: {auc:.4f}"
    )

    print(
        f"Brier score: {brier:.4f}"
    )

    # -----------------------------------------
    # State-level Actual vs Expected
    # -----------------------------------------

    data["expected_variance"] = (
        data["expected_probability"]
        * (
            1
            - data[
                "expected_probability"
            ]
        )
    )

    diagnostics = (
        data.groupby(
            "state_code"
        )
        .agg(
            orders=(
                "negative_review",
                "size",
            ),
            actual_negative_reviews=(
                "negative_review",
                "sum",
            ),
            expected_negative_reviews=(
                "expected_probability",
                "sum",
            ),
            actual_negative_review_rate=(
                "negative_review",
                "mean",
            ),
            expected_negative_review_rate=(
                "expected_probability",
                "mean",
            ),
            expected_variance=(
                "expected_variance",
                "sum",
            ),
        )
        .reset_index()
    )

    diagnostics["residual_rate"] = (
        diagnostics[
            "actual_negative_review_rate"
        ]
        - diagnostics[
            "expected_negative_review_rate"
        ]
    )

    diagnostics[
        "residual_pp"
    ] = (
        diagnostics[
            "residual_rate"
        ]
        * 100
    )

    diagnostics["residual_se"] = (
        np.sqrt(
            diagnostics[
                "expected_variance"
            ]
        )
        / diagnostics["orders"]
    )

    diagnostics["ci_lower_pp"] = (
        (
            diagnostics[
                "residual_rate"
            ]
            - 1.96
            * diagnostics[
                "residual_se"
            ]
        )
        * 100
    )

    diagnostics["ci_upper_pp"] = (
        (
            diagnostics[
                "residual_rate"
            ]
            + 1.96
            * diagnostics[
                "residual_se"
            ]
        )
        * 100
    )

    diagnostics["z_score"] = (
        (
            diagnostics[
                "actual_negative_reviews"
            ]
            - diagnostics[
                "expected_negative_reviews"
            ]
        )
        / np.sqrt(
            diagnostics[
                "expected_variance"
            ]
        )
    )

    diagnostics["diagnostic_state"] = (
        diagnostics.apply(
            classify_diagnostic,
            axis=1,
        )
    )

    diagnostics = (
        diagnostics.sort_values(
            "z_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    serving = diagnostics[
        [
            "state_code",
            "orders",
            "actual_negative_review_rate",
            "expected_negative_review_rate",
            "residual_pp",
            "ci_lower_pp",
            "ci_upper_pp",
            "z_score",
            "diagnostic_state",
        ]
    ].copy()

    serving = serving.rename(
        columns={
            "orders": "evidence_count",
        }
    )

    serving["model_version"] = (
        MODEL_VERSION
    )

    serving["generated_at"] = (
        datetime.now(timezone.utc)
    )


    expected_states = 27

    allowed_diagnostic_states = {
        "WORSE_THAN_EXPECTED",
        "BETTER_THAN_EXPECTED",
        "AS_EXPECTED",
        "INSUFFICIENT_EVIDENCE",
    }

    if len(serving) != expected_states:
        raise ValueError(
            "Expected exactly 27 Brazilian states, "
            f"got {len(serving)}."
        )

    if serving["state_code"].nunique() != expected_states:
        raise ValueError(
            "State codes must be unique."
        )

    if serving.isna().any().any():
        raise ValueError(
            "Serving diagnostics contain null values."
        )

    invalid_states = (
        set(serving["diagnostic_state"])
        - allowed_diagnostic_states
    )

    if invalid_states:
        raise ValueError(
            "Unexpected diagnostic states: "
            f"{sorted(invalid_states)}"
        )

    job_config = (
        bigquery.LoadJobConfig(
            write_disposition=(
                bigquery.WriteDisposition.WRITE_TRUNCATE
            ),
            create_disposition=(
                bigquery.CreateDisposition.CREATE_NEVER
            ),
            schema=[
                bigquery.SchemaField(
                    "state_code",
                    "STRING",
                    mode="REQUIRED",
                ),
                bigquery.SchemaField(
                    "evidence_count",
                    "INTEGER",
                    mode="REQUIRED",
                ),
                bigquery.SchemaField(
                    "actual_negative_review_rate",
                    "FLOAT",
                    mode="REQUIRED",
                ),
                bigquery.SchemaField(
                    "expected_negative_review_rate",
                    "FLOAT",
                    mode="REQUIRED",
                ),
                bigquery.SchemaField(
                    "residual_pp",
                    "FLOAT",
                    mode="REQUIRED",
                ),
                bigquery.SchemaField(
                    "ci_lower_pp",
                    "FLOAT",
                    mode="REQUIRED",
                ),
                bigquery.SchemaField(
                    "ci_upper_pp",
                    "FLOAT",
                    mode="REQUIRED",
                ),
                bigquery.SchemaField(
                    "z_score",
                    "FLOAT",
                    mode="REQUIRED",
                ),
                bigquery.SchemaField(
                    "diagnostic_state",
                    "STRING",
                    mode="REQUIRED",
                ),
                bigquery.SchemaField(
                    "model_version",
                    "STRING",
                    mode="REQUIRED",
                ),
                bigquery.SchemaField(
                    "generated_at",
                    "TIMESTAMP",
                    mode="REQUIRED",
                ),
            ],
        )
    )

    load_job = (
        client.load_table_from_dataframe(
            serving,
            TABLE_ID,
            job_config=job_config,
        )
    )

    load_job.result()

    print()
    print(
        f"Persisted {len(serving)} states "
        f"to {TABLE_ID}"
    )

    display_columns = [
        "state_code",
        "orders",
        "diagnostic_state",
        "actual_negative_review_rate",
        "expected_negative_review_rate",
        "residual_pp",
        "ci_lower_pp",
        "ci_upper_pp",
        "z_score",
    ]

    print()
    print(
        "=== STATE ACTUAL VS EXPECTED ==="
    )

    print(
        diagnostics[
            display_columns
        ]
        .round(
            {
                "actual_negative_review_rate": 4,
                "expected_negative_review_rate": 4,
                "residual_pp": 2,
                "ci_lower_pp": 2,
                "ci_upper_pp": 2,
                "z_score": 2,
            }
        )
        .to_string(
            index=False
        )
    )

    diagnostics.to_csv(
        OUTPUT_DIR
        / "business_decision_v2_state_diagnostics.csv",
        index=False,
    )

    print()
    print(
        "State diagnostics written to "
        "analysis/output/"
        "business_decision_v2_state_diagnostics.csv"
    )


if __name__ == "__main__":
    main()