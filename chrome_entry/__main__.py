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
    try:
        out_msg = '{"Suggestions": [], "Error": "Model loaded, running predictions."}'
        send_message(out_msg)
        local_server = xmlrpc.client.ServerProxy(SERVER_ADDR, allow_none=True)
        if pr_id:
            suggestions = local_server.predict_pr(repo, pr_id)
        elif issue_id:
            suggestions = local_server.predict_issue(repo, issue_id)
        if len(suggestions) > 0:
            out_msg = '{"Suggestions": %s, "Error": ""}' \
                      % json.dumps([{'Id': p[0], 'Title': p[1], 'Probability': float('%.2f' % p[2]),
                                     'Repo': msg['Repository']} for p in suggestions])
        else:
            out_msg = '{"Suggestions": [], "Error": "No suggestions available"}'
        send_message(out_msg)
    except Exception as e:
        if 'RateLimitExceededException' in str(e):
            out_msg = '{"Suggestions": [], "Error": "Github API rate-limit exceeded"}'
            send_message(out_msg)
        else:
            raise e


def unknown_type(msg):
    out_msg = '{"Suggestions": [], "Error": "Unknown message type: %s"}' % msg['Type']
    send_message(out_msg)


def model_update():
    local_server = xmlrpc.client.ServerProxy(SERVER_ADDR, allow_none=True)
    local_server.trigger_model_updates()
    out_msg = '{"Suggestions": [], "Error": "Updated model, please close and reopen plugin pop-up to use!"}'
    send_message(out_msg)


def record_links(msg):
    repo = msg['Repository'].translate({ord(c): '_' for c in '\\/'})
    local_server = xmlrpc.client.ServerProxy(SERVER_ADDR, allow_none=True)
    for issue_id, pr_id in json.loads(msg['Links']):
        local_server.record_link(repo, issue_id, pr_id)
    out_msg = '{"Suggestions": [], "Error": "Links recorded, please close and reopen plugin pop-up to use!"}'
    send_message(out_msg)


def dummy():
    """
    Dummy function used as a stub to debug w/o running the backend. Reuses a prerecorded message.
    :return: None
    """
    with open('previous_suggestion.txt') as f:
        out_msg = f.read()
    send_message(out_msg)


# Thread that reads messages from the webapp.
def threshold_request(msg):
    repo = msg['Repository'].translate({ord(c): '_' for c in '\\/'})
    local_server = xmlrpc.client.ServerProxy(SERVER_ADDR, allow_none=True)
    threshold = local_server.request_mean_threshold(repo)
    out_msg = '{"Threshold": %2.2f}' % threshold
    send_message(out_msg)


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
            out_msg = '{"Suggestions": [], "Error": "Malformed JSON message received!"}'
            send_message(out_msg)
            continue
        try:
            if msg['Type'] == 'Prediction':
                process_prediction_request(msg)
            elif msg['Type'] == 'Update':
                model_update()
            elif msg['Type'] == 'LinkUpdate':
                record_links(msg)
            elif msg['Type'] == 'Threshold':
                threshold_request(msg)
            else:
                unknown_type(msg)
        except Exception as e:
            out_msg = '{"Suggestions": [], "Error": "Failed due to %s!"}' % str(e).replace('"', '\'\'')
            send_message(out_msg)
            continue


def main():
    read_thread_func()
    sys.exit(0)


if __name__ == '__main__':
    main()
