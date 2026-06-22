import os
import urllib.request
import urllib.parse
import mimetypes

def test_health():
    try:
        response = urllib.request.urlopen("http://localhost:8000/health")
        content = response.read().decode('utf-8')
        print("Health Check passed!")
        print("Response:", content)
    except Exception as e:
        print("Health Check failed:", e)

def test_upload():
    # Construct multipart/form-data payload without external libraries
    url = "http://localhost:8000/upload"
    filepath = "test_dataset.csv"
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    if not os.path.exists(filepath):
        print(f"Test dataset not found at {filepath}")
        return

    with open(filepath, 'rb') as f:
        file_content = f.read()

    filename = os.path.basename(filepath)
    content_type = mimetypes.guess_type(filepath)[0] or 'application/octet-stream'

    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode('utf-8') + file_content + f"\r\n--{boundary}--\r\n".encode('utf-8')

    req = urllib.request.Request(url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    req.add_header('Content-Length', len(body))

    try:
        response = urllib.request.urlopen(req)
        response_content = response.read().decode('utf-8')
        print("Upload test passed!")
        import json
        parsed = json.loads(response_content)
        print("JSON Response:")
        print(json.dumps(parsed, indent=2))
    except Exception as e:
        print("Upload test failed:", e)
        if hasattr(e, 'read'):
            print("Error details:", e.read().decode('utf-8'))

if __name__ == "__main__":
    test_health()
    print("-" * 40)
    test_upload()
