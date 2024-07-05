import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

import json
import requests
import time
import sys

client_id = os.getenv("RT_CLIENT_ID")
client_secret = os.getenv("RT_CLIENT_SECRET")
transcribe_file = sys.argv[1]
print(f"transcribe_file: {transcribe_file}")

# Measure time for authentication request
start_time = time.time()
resp = requests.post(
    'https://openapi.vito.ai/v1/authenticate',
    data={
        'client_id': client_id,
        'client_secret': client_secret
    }
)
end_time = time.time()
print(f"Authentication request took {end_time - start_time:.2f} seconds")

resp.raise_for_status()
accessToken = resp.json()['access_token']
print(f"Access Token: {accessToken}")

config = {
            'domain': 'CALL',
            "use_diarization": False,
            "use_itn": True,
            "use_disfluency_filter": False,
            "use_profanity_filter": False,
            "use_paragraph_splitter": False,
            "paragraph_splitter": {"max": 50}
}


# Measure time for transcription request submission
start_time = time.time()
resp = requests.post(
    'https://openapi.vito.ai/v1/transcribe',
    headers={'Authorization': 'bearer ' + accessToken},
    data={'config': json.dumps(config)},
    files={'file': open(transcribe_file, 'rb')}
)
end_time = time.time()
print(f"Transcription request submission took {end_time - start_time:.2f} seconds")

resp.raise_for_status()
transcription_id = resp.json()['id']
print(f"Transcription ID: {transcription_id}")

# Measure total time for transcription processing
processing_start_time = time.time()

# Check the status of the transcription
status_url = f'https://openapi.vito.ai/v1/transcribe/{transcription_id}'
while True:
    start_time = time.time()
    status_resp = requests.get(
        status_url,
        headers={'Authorization': 'bearer ' + accessToken}
    )
    end_time = time.time()
    print(f"Status request took {end_time - start_time:.2f} seconds")

    status_resp.raise_for_status()
    status_data = status_resp.json()
    # print(status_data)

    if status_data['status'] == 'completed':
        processing_end_time = time.time()
        print("Transcription completed.")
        print(status_data['results'])
        break
    elif status_data['status'] == 'failed':
        print("Transcription failed.")
        break
    else:
        print("Transcription in progress. Waiting for 5 seconds before checking again...")
        time.sleep(5)

total_processing_time = processing_end_time - processing_start_time
print(f"Total transcription processing time: {total_processing_time:.2f} seconds")

# Extracting and formatting the data
utterances = status_data['results']['utterances']
combined_msg = ' '.join(f"{utterance['msg']}" for utterance in utterances)

# Saving the formatted message to a file
output_file = "stt/stt.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(combined_msg)

print(f"Transcription result saved to {output_file}")
