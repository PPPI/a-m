import json
import struct
import sys
import xmlrpc.client

SERVER_ADDR = 'http://localhost:8000'
# Helper function that sends a message to the webapp.
def send_message(message):
    # Write message size.
    sys.stdout.buffer.write(struct.pack('i', len(message)))
    # Write the message itself.
    sys.stdout.buffer.write(message.encode('utf-8'))
    sys.stdout.buffer.flush()


def process_prediction_request(msg):
    repo = msg['Repository'].translate({ord(c): '_' for c in '\\/'})
    pr_id = msg['PR']
    issue_id = msg['Issue']
    threshold = msg['Threshold']
    out_msg = '{"Suggestions": [], "Error": "Received data, loading model."}'
    send_message(out_msg)

    try:
        out_msg = '{"Suggestions": [], "Error": "Model loaded, running predictions."}'
        send_message(out_msg)
        local_server = xmlrpc.client.ServerProxy(SERVER_ADDR)
        if pr_id:
            suggestions = local_server.predict_pr(repo, pr_id)
        elif issue_id:
            suggestions = local_server.predict_issue(repo, issue_id)
        with open('debug_.txt', 'w') as f:
            f.write('Got suggestions: %s' % str(suggestions))
        suggestions = [s for s in suggestions if s[-1] >= threshold]
        with open('debug.txt', 'w') as f:
            f.write('Got suggestions: %s' % str(suggestions))
        if len(suggestions) > 0:
            out_msg = '{"Suggestions": %s, "Error": ""}' \
                      % json.dumps([{'Id': p[0], 'Title': p[1], 'Probability': float('%.2f' % p[2]),
                                     'Repo': msg['Repository']} for p in suggestions])
            with open('debug.txt', 'w') as f:
                f.write(out_msg)
        else:
            out_msg = '{"Suggestions": [], "Error": "No suggestions available"}'
        send_message(out_msg)
    except Exception as e:
        with open('debug.txt', 'w') as f:
            f.write('Failed Predict, %s' % str(e))
        if 'RateLimitExceededException' in str(e):
            out_msg = '{"Suggestions": [], "Error": "Github API rate-limit exceeded"}'
        else:
            out_msg = '{"Suggestions": [], "Error": "Failed to generate suggestions due to %s"}' % json.dumps(str(e))
        send_message(out_msg)


def unknown_type(msg):
    out_msg = '{"Suggestions": [], "Error": "Unknown message type: %s"}' % msg['Type']
    send_message(out_msg)


def model_update():
    local_server = xmlrpc.client.ServerProxy(SERVER_ADDR)
    local_server.trigger_model_updates()
    out_msg = '{"Suggestions": [], "Error": "Updated model"}'
    send_message(out_msg)


def record_links(msg):
    # TODO: Implement recording links
    local_server = xmlrpc.client.ServerProxy(SERVER_ADDR)
    for link in msg['Links']:
        local_server.record_link(*link)


# Thread that reads messages from the webapp.
def read_thread_func():
    while True:
        # Read the message length (first 4 bytes).
        text_length_bytes = sys.stdin.buffer.read(4)
        if len(text_length_bytes) == 0:
            sys.exit(0)
        # Unpack message length as 4 byte integer.
        text_length = struct.unpack('i', text_length_bytes)[0]
        # Read the text (JSON object) of the message.
        text = sys.stdin.buffer.read(text_length).decode('utf-8')
        try:
            msg = json.loads(text)
        except Exception as e:
            # with open('debug.txt', 'w') as f:
            #     f.write('Failed JSON, %s' % str(e))
            out_msg = '{"Suggestions": [], "Error": "Malformed JSON message received!"}'
            send_message(out_msg)
            continue
        if msg['Type'] == 'Prediction':
            process_prediction_request(msg)
        elif msg['Type'] == 'Update':
            model_update()
        elif msg['Type'] == 'LinkUpdate':
            record_links(msg)
        else:
            unknown_type(msg)


def main():
    read_thread_func()
    sys.exit(0)


if __name__ == '__main__':
    main()
