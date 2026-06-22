"""Quick test of TCIA API connectivity."""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=5, backoff_factor=3)
session.mount("https://", HTTPAdapter(max_retries=retry))

print("Testing TCIA API connectivity...")
try:
    resp = session.get(
        "https://services.cancerimagingarchive.net/services/v4/TCIA/query/getPatient?collection=LIDC-IDRI",
        timeout=120,
    )
    print(f"Status: {resp.status_code}")
    print(f"Content length: {len(resp.text)}")
    patients = resp.json()
    print(f"Number of patients: {len(patients)}")
    first_ids = [p["PatientID"] for p in patients[:5]]
    print(f"First 5 patient IDs: {first_ids}")
except Exception as e:
    print(f"ERROR: {e}")