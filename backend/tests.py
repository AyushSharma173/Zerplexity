import boto3, json, datetime, sys, os

client   = boto3.client("sagemaker-runtime", region_name="us-east-2")
ENDPOINT = "Endpoint-20250717-165210"

def ts():
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

payload = {
    "inputs": "Hello! How are you doing today? Here is a 2000 word story",
    "parameters": {"max_new_tokens": 1000, "temperature": 0.7,
                   "top_p": 0.9, "stream": True}
}

print("SDK version check →", boto3.__version__, "/", client.meta.config.user_agent)

print(f"[{ts()}] sending request …", flush=True)

resp = client.invoke_endpoint_with_response_stream(
    EndpointName = ENDPOINT,
    ContentType  = "application/json",
    Accept       = "application/x-text",   # 👈 plain text, no JSON to parse
    Body         = json.dumps(payload),
)

for event in resp["Body"]:
    if "PayloadPart" in event:
        sys.stdout.write(event["PayloadPart"]["Bytes"].decode("utf‑8"))
        sys.stdout.flush()