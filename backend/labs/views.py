
import json
import datetime  
from django.http import JsonResponse  # ✅ Ensures proper HTTP response handling
from django.views.decorators.csrf import csrf_exempt  # ✅ (Optional) To allow API testing

# Manually created database of 50 valid licenses with random numbers
VALID_LICENSES = {
    "LAB734218": {"issued_by": "Medical Council", "expires_on": "2026-12-31"},
    "LAB289354": {"issued_by": "Health Department", "expires_on": "2025-06-30"},
    "LAB918472": {"issued_by": "National Lab Authority", "expires_on": "2027-09-15"},
    "LAB567321": {"issued_by": "Medical Council", "expires_on": "2028-03-20"},
    "LAB450923": {"issued_by": "Health Department", "expires_on": "2029-01-10"},
    "LAB678412": {"issued_by": "National Lab Authority", "expires_on": "2026-07-05"},
    "LAB892341": {"issued_by": "Medical Council", "expires_on": "2027-11-25"},
    "LAB125678": {"issued_by": "Health Department", "expires_on": "2028-08-18"},
    "LAB903214": {"issued_by": "National Lab Authority", "expires_on": "2025-04-12"},
    "LAB672389": {"issued_by": "Medical Council", "expires_on": "2027-02-07"},
    "LAB748210": {"issued_by": "Health Department", "expires_on": "2029-09-30"},
    "LAB561289": {"issued_by": "National Lab Authority", "expires_on": "2026-03-14"},
    "LAB984512": {"issued_by": "Medical Council", "expires_on": "2028-05-22"},
    "LAB234786": {"issued_by": "Health Department", "expires_on": "2027-07-19"},
    "LAB876543": {"issued_by": "National Lab Authority", "expires_on": "2026-11-10"},
    "LAB982143": {"issued_by": "Medical Council", "expires_on": "2028-10-05"},
    "LAB341209": {"issued_by": "Health Department", "expires_on": "2025-12-12"},
    "LAB562738": {"issued_by": "National Lab Authority", "expires_on": "2027-04-14"},
    "LAB874321": {"issued_by": "Medical Council", "expires_on": "2028-09-28"},
    "LAB492108": {"issued_by": "Health Department", "expires_on": "2026-06-20"},
    "LAB239875": {"issued_by": "National Lab Authority", "expires_on": "2029-03-16"},
    "LAB598761": {"issued_by": "Medical Council", "expires_on": "2027-01-23"},
    "LAB689432": {"issued_by": "Health Department", "expires_on": "2025-05-09"},
    "LAB123987": {"issued_by": "National Lab Authority", "expires_on": "2026-08-14"},
    "LAB839210": {"issued_by": "Medical Council", "expires_on": "2029-07-11"},
    "LAB982374": {"issued_by": "Health Department", "expires_on": "2028-04-19"},
    "LAB472389": {"issued_by": "National Lab Authority", "expires_on": "2025-10-31"},
    "LAB782943": {"issued_by": "Medical Council", "expires_on": "2027-05-20"},
    "LAB982134": {"issued_by": "Health Department", "expires_on": "2026-02-25"},
    "LAB289473": {"issued_by": "National Lab Authority", "expires_on": "2028-06-30"},
    "LAB571893": {"issued_by": "Medical Council", "expires_on": "2025-03-21"},
    "LAB798432": {"issued_by": "Health Department", "expires_on": "2027-08-29"},
    "LAB123678": {"issued_by": "National Lab Authority", "expires_on": "2026-01-07"},
    "LAB983476": {"issued_by": "Medical Council", "expires_on": "2029-02-18"},
    "LAB564893": {"issued_by": "Health Department", "expires_on": "2028-09-09"},
    "LAB412309": {"issued_by": "National Lab Authority", "expires_on": "2027-12-31"},
    "LAB237891": {"issued_by": "Medical Council", "expires_on": "2025-07-08"},
    "LAB783294": {"issued_by": "Health Department", "expires_on": "2026-05-14"},
    "LAB198342": {"issued_by": "National Lab Authority", "expires_on": "2029-08-22"},
    "LAB492761": {"issued_by": "Medical Council", "expires_on": "2028-02-11"},
    "LAB673984": {"issued_by": "Health Department", "expires_on": "2027-09-16"},
    "LAB139872": {"issued_by": "National Lab Authority", "expires_on": "2026-10-12"},
    "LAB892731": {"issued_by": "Medical Council", "expires_on": "2029-05-07"},
    "LAB562734": {"issued_by": "Health Department", "expires_on": "2028-12-20"},
    "LAB198473": {"issued_by": "National Lab Authority", "expires_on": "2027-06-03"},
    "LAB672184": {"issued_by": "Medical Council", "expires_on": "2026-03-28"},
    "LAB348291": {"issued_by": "Health Department", "expires_on": "2025-08-15"},
    "LAB764291": {"issued_by": "National Lab Authority", "expires_on": "2029-10-01"},
    "LAB985621": {"issued_by": "Medical Council", "expires_on": "2028-04-30"},
    "LAB372894": {"issued_by": "Health Department", "expires_on": "2027-11-22"},
}

@csrf_exempt  
def validate_license(request):
    if request.method != "POST":
        return JsonResponse({"status": "Error", "message": "Only POST requests allowed"}, status=405)

    try:
        data = json.loads(request.body)
        license_number = data.get("license_number", "").strip().upper()
    except json.JSONDecodeError:
        return JsonResponse({"status": "Error", "message": "Invalid JSON format"}, status=400)

    if not license_number:
        return JsonResponse({"status": "Invalid", "message": "License number is required."}, status=400)

    license_data = VALID_LICENSES.get(license_number)

    if not license_data:
        return JsonResponse({"status": "Invalid", "message": "License number not found."}, status=404)

    return JsonResponse({"status": "Valid", "message": "License is valid."})


