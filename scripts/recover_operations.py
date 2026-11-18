import argparse
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or recover interrupted studio operations.")
    parser.add_argument("--db", type=Path, required=True, help="Path to the app.sqlite in use.")
    parser.add_argument("--api-stopped", action="store_true", help="Confirm all API processes are stopped.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resolve-operation", nargs=2, metavar=("SCOPE", "KEY"))
    action.add_argument("--release-asset", nargs=2, metavar=("ASSET_ID", "OWNER_TOKEN"))
    args = parser.parse_args()

    changing = args.resolve_operation is not None or args.release_asset is not None
    if changing and not args.api_stopped:
        parser.error("Stop all API processes and inspect the affected project before using --api-stopped.")

    mode = "rw" if changing else "ro"
    db_uri = f"{args.db.resolve().as_uri()}?mode={mode}"
    with closing(sqlite3.connect(db_uri, uri=True, timeout=10)) as connection:
        with connection:
            if args.resolve_operation is not None:
                scope, key = args.resolve_operation
                outcome = {
                    "error": {
                        "status_code": 409,
                        "detail": "The API stopped before recording the result. "
                                  "Check the project or asset before starting another attempt.",
                        "headers": {"Idempotency-Status": "uncertain"},
                    },
                }
                cursor = connection.execute(
                    """
                    UPDATE operation_receipts
                    SET status = 'uncertain', outcome_json = ?, updated_at = ?
                    WHERE scope = ? AND key = ? AND status = 'running'
                    """,
                    (json.dumps(outcome), datetime.now(timezone.utc).isoformat(), scope, key),
                )
                if cursor.rowcount != 1:
                    parser.error("No running receipt matches that scope and key.")
                print("Receipt marked uncertain. Its original key remains reserved.")
            elif args.release_asset is not None:
                asset_id, owner_token = args.release_asset
                cursor = connection.execute(
                    """
                    DELETE FROM asset_generation_claims
                    WHERE asset_id = ? AND owner_token = ?
                    """,
                    (asset_id, owner_token),
                )
                if cursor.rowcount != 1:
                    parser.error("No asset claim matches that asset ID and owner token.")
                print("Asset claim released. No generation was started.")
            else:
                connection.row_factory = sqlite3.Row
                operations = connection.execute(
                    """
                    SELECT scope, key, created_at
                    FROM operation_receipts WHERE status = 'running'
                    ORDER BY created_at
                    """
                ).fetchall()
                claims = connection.execute(
                    """
                    SELECT asset_id, owner_token, created_at
                    FROM asset_generation_claims ORDER BY created_at
                    """
                ).fetchall()
                print(json.dumps({
                    "running_operations": [dict(row) for row in operations],
                    "asset_claims": [dict(row) for row in claims],
                }, indent=2))


if __name__ == "__main__":
    main()