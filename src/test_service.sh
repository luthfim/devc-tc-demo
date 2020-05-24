header="Content-Type: application/json"
req_type="POST"
data='{"text":["hey check out my new channel", "Eminem is literally the best rapper in history"]}'
socket="http://localhost:5000/predict"

echo "Input: $data"
curl -X POST --header "$header" --data "$data" $socket
