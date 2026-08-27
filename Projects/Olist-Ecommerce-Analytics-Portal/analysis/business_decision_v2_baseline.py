import numpy as np
import pandas as pd
import statsmodels.api as sm

from google.cloud import bigquery
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
)


PROJECT_ID = "balmy-nuance-468118-g4"


QUERY = """
SELECT
  IF(
    avg_review_score <= 2,
    1,
    0
  ) AS negative_review,

  IF(
    is_late_delivery,
    1,
    0
  ) AS is_late_delivery,

  days_purchase_to_customer_delivery
    AS delivery_days,

  order_gross_value,

  IF(
    order_total_items > 1,
    1,
    0
  ) AS multi_item_order

FROM
  `balmy-nuance-468118-g4.olist_marts.fct_orders`

WHERE
  avg_review_score IS NOT NULL
  AND order_delivered_customer_date
    IS NOT NULL
  AND order_estimated_delivery_date
    IS NOT NULL
  AND days_purchase_to_customer_delivery
    IS NOT NULL
"""


def main() -> None:
    client = bigquery.Client(
        project=PROJECT_ID
    )

    print("Loading model data...")

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

    features = [
        "is_late_delivery",
        "delivery_days_log",
        "gmv_log",
        "multi_item_order",
    ]

    X = data[features].astype(float)
    y = data["negative_review"].astype(int)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_model = sm.add_constant(
        X_train
    )

    X_test_model = sm.add_constant(
        X_test,
        has_constant="add",
    )

    model = sm.Logit(
        y_train,
        X_train_model,
    )

    result = model.fit(
        disp=False
    )

    print()
    print("=== MODEL SUMMARY ===")
    print(result.summary())

    coefficients = pd.DataFrame(
        {
            "coefficient":
                result.params,
            "odds_ratio":
                np.exp(result.params),
            "p_value":
                result.pvalues,
        }
    )

    print()
    print(
        "=== COEFFICIENTS / ODDS RATIOS ==="
    )
    print(
        coefficients.round(4)
    )

    probabilities = result.predict(
        X_test_model
    )

    calibration = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": probabilities.to_numpy(),
        }
    )

    calibration["risk_decile"] = pd.qcut(
        calibration["predicted"],
        q=10,
        duplicates="drop",
    )

    calibration_summary = (
        calibration.groupby(
            "risk_decile",
            observed=True,
        )
        .agg(
            orders=("actual", "size"),
            mean_predicted_probability=(
                "predicted",
                "mean",
            ),
            actual_negative_review_rate=(
                "actual",
                "mean",
            ),
        )
        .reset_index()
    )

    calibration_summary["gap_pp"] = (
        (
            calibration_summary[
                "actual_negative_review_rate"
            ]
            - calibration_summary[
                "mean_predicted_probability"
            ]
        )
        * 100
    )

    print()
    print("=== CALIBRATION BY RISK DECILE ===")
    print(
        calibration_summary.to_string(
            index=False
        )
    )

    ece = np.average(
        np.abs(
            calibration_summary["gap_pp"]
        ),
        weights=calibration_summary["orders"],
    )

    print()
    print(
        "Weighted mean absolute calibration "
        f"gap: {ece:.2f} percentage points"
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    print()
    print("=== TEST PERFORMANCE ===")
    print(
        f"Test rows: {len(y_test):,}"
    )
    print(
        f"Actual negative-review rate: "
        f"{y_test.mean():.4f}"
    )
    print(
        f"Mean predicted probability: "
        f"{probabilities.mean():.4f}"
    )
    print(
        f"ROC AUC: {auc:.4f}"
    )
    print(
        f"Brier score: {brier:.4f}"
    )


if __name__ == "__main__":
    main()