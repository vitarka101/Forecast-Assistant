import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.db.session import SessionLocal, init_db
from app.schemas.requests import TrainPipelineRequest
from app.services.pipeline import TrainingPipelineService


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the retail router-executor pipeline.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--product-clusters", type=int, default=4)
    parser.add_argument("--customer-clusters", type=int, default=4)
    parser.add_argument("--min-history-weeks", type=int, default=8)
    parser.add_argument("--max-entities-for-dtw", type=int, default=250)
    parser.add_argument("--lag-weeks", type=int, default=8)
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        response = TrainingPipelineService(db).run(
            TrainPipelineRequest(
                data_path=args.data_path,
                product_clusters=args.product_clusters,
                customer_clusters=args.customer_clusters,
                min_history_weeks=args.min_history_weeks,
                max_entities_for_dtw=args.max_entities_for_dtw,
                lag_weeks=args.lag_weeks,
            )
        )
        print(response.model_dump_json(indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()

