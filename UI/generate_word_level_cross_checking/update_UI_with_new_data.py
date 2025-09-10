import os
import json
import requests
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()
LS_URL = os.getenv("LS_URL", "http://localhost:8080")
API_TOKEN = os.getenv("API_TOKEN")
PROJECT_ID = int(os.getenv("PROJECT_ID", "0"))

HEADERS = {"Authorization": f"Token {API_TOKEN}", "Content-Type": "application/json"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# --- helpers ---------------------------------------------------------------

def _extract_results(payload):
    """Support both list returns and dict-with-results pagination."""
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    return []

def _next_link(payload):
    if isinstance(payload, dict):
        return payload.get("next")
    return None

def _extract_batch(payload):
    """Normalize payload into a list of items."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], list):
            return payload["results"]
        if "tasks" in payload and isinstance(payload["tasks"], list):
            return payload["tasks"]
    return []

def fetch_all(url, params=None, start_page=1, page_param="page", id_field="id"):
    """
    Iterate pages until 404, empty batch, or a duplicate item is found.
    Duplicates are detected by id_field if present, else by whole object.
    """
    items = []
    seen_ids = set()
    base_params = dict(params or {})
    page = start_page
    stop = False

    while not stop:
        q = {**base_params, page_param: page}
        resp = SESSION.get(url, params=q, timeout=60)

        if resp.status_code == 404:
            break

        resp.raise_for_status()
        payload = resp.json()
        batch = _extract_batch(payload)

        if not batch:
            break

        for item in batch:
            # Use id_field if available, else fallback to full object tuple
            item_id = item.get(id_field) if isinstance(item, dict) else tuple(sorted(item.items())) if isinstance(item, dict) else item
            if item_id in seen_ids:
                stop = True
                break
            seen_ids.add(item_id)
            items.append(item)

        page += 1

    return items

def list_project_tasks(project_id):
    url = f"{LS_URL}/api/projects/{project_id}/tasks"
    # fields hint is safe even if server ignores it
    return fetch_all(url, params={"page_size": 100, "fields": "id"})

def list_task_predictions(task_id):
    url = f"{LS_URL}/api/predictions"
    return fetch_all(url, params={"task": task_id, "page_size": 100})

def delete_prediction(pred_id):
    url = f"{LS_URL}/api/predictions/{pred_id}"
    r = SESSION.delete(url, timeout=60)
    r.raise_for_status()

def post_prediction(task_id, pred):
    """Repost a cleaned prediction for the task."""
    url = f"{LS_URL}/api/predictions"
    payload = {
        "task": task_id,
        "result": pred.get("result", []),
        "model_version": pred.get("model_version", "preannotation_cleaned"),
    }
    if "score" in pred:
        payload["score"] = pred["score"]
    r = SESSION.post(url, data=json.dumps(payload), timeout=60)
    r.raise_for_status()

def clean_results(results, comment_from=("issues",), comment_to=("commentHeader",)):
    """Remove comment results; return (filtered_results, changed_bool)."""
    filtered = []
    changed = False
    for r in results:
        if r.get("from_name") in comment_from:
            changed = True
            continue
        if r.get("to_name") in comment_to:
            changed = True
            continue
        filtered.append(r)
    return filtered, changed

# --- main ------------------------------------------------------------------

def main():
    if not API_TOKEN:
        raise RuntimeError("API_TOKEN is not set (.env)")

    total_tasks = 0
    updated_tasks = 0
    total_preds_changed = 0

    tasks = list_project_tasks(PROJECT_ID)
    for t in tqdm(tasks):
        task_id = t["id"]
        total_tasks += 1

        preds = list_task_predictions(task_id)
        any_change_for_task = False

        for p in preds:
            original_results = p.get("result", [])
            cleaned_results, changed = clean_results(original_results)

            if not changed:
                continue  # nothing to do for this prediction

            # Replace: delete old prediction, post cleaned copy
            delete_prediction(p["id"])
            cleaned_pred_payload = {
                "result": cleaned_results,
                "model_version": p.get("model_version", "preannotation_cleaned"),
            }
            post_prediction(task_id, cleaned_pred_payload)

            any_change_for_task = True
            total_preds_changed += 1

        if any_change_for_task:
            updated_tasks += 1
            print(f"Updated task id={task_id}")

    print(f"Done. Tasks scanned: {total_tasks}, tasks updated: {updated_tasks}, predictions replaced: {total_preds_changed}")

if __name__ == "__main__":
    main()
