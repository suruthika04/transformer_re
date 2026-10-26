import urllib.request
import json

BASE_URL = "http://127.0.0.1:8000/api"

def make_post_request(url, data_dict):
    req = urllib.request.Request(
        url,
        data=json.dumps(data_dict).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def make_get_request(url):
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode('utf-8'))

def test_api():
    print("1. Testing /api/health...")
    h = make_get_request(f"{BASE_URL}/health")
    print("Health response:", h)

    print("\n2. Testing /api/evaluate with example sentence...")
    payload = {
        "text": "The cat is sleeping on the mat because it is tired.",
        "threshold": 0.50
    }
    res = make_post_request(f"{BASE_URL}/evaluate", payload)
    
    print("\n[EVALUATE RESPONSE SUMMARY]")
    print(f"Original Tokens: {res['original_tokens']}")
    print(f"Selected Tokens: {res['selected_tokens']}")
    print(f"Tokens Saved: {res['tokens_saved']}")
    print(f"Saving Percentage: {res['saving_percentage']}%")
    print(f"Retained: {res['retained_tokens']}")
    print(f"Filtered: {res['filtered_tokens']}")
    print(f"Chatbot Response: {res['chatbot_response']}")
    
    print("\n3. Testing /api/recalculate with threshold=0.70...")
    recalc_payload = {
        "token_analysis": res["token_analysis"],
        "threshold": 0.70
    }
    recalc_res = make_post_request(f"{BASE_URL}/recalculate", recalc_payload)
    print(f"New Retained (0.70): {recalc_res['retained_tokens']}")
    print(f"New Tokens Saved: {recalc_res['tokens_saved']}")
    print(f"New Saving %: {recalc_res['saving_percentage']}%")

    print("\n[SUCCESS] API ENDPOINTS WORKING PERFECTLY!")

if __name__ == "__main__":
    test_api()
