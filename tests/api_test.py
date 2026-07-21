import requests

def test_verify_health():
    try:
        res = requests.get("http://localhost:8000/")
        if res.status_code==200:
            data = res.json()
            assert data['success']
        else:
            print("Failed",res.status_code)
    except requests.exceptions.RequestException as e:
        print("Error Occuured:",e)


