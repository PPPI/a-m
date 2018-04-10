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
    try:
        with open('previous_ids.txt') as f:
            lines = [line for line in f.readline()]
        assert len(lines) == 3
        previous_repo = lines[0].strip()
        previous_pr_id = lines[1].strip()
        previous_issue_id = lines[2].strip()
    except (FileNotFoundError, AssertionError):
        previous_repo = ''
        previous_pr_id = ''
        previous_issue_id = ''

    repo = msg['Repository'].translate({ord(c): '_' for c in '\\/'})
    pr_id = msg['PR']
    issue_id = msg['Issue']
    if repo == previous_repo and pr_id == previous_pr_id and issue_id == previous_issue_id:
        with open('previous_suggestion.txt') as f:
            out_msg = f.read()
        send_message(out_msg)
        return
    try:
        out_msg = '{"Suggestions": [], "Error": "Model loaded, running predictions."}'
        send_message(out_msg)
        local_server = xmlrpc.client.ServerProxy(SERVER_ADDR)
        if pr_id:
            suggestions = local_server.predict_pr(repo, pr_id)
        elif issue_id:
            suggestions = local_server.predict_issue(repo, issue_id)
        # with open('debug.txt', 'w') as f:
        #     f.write('Got suggestions: %s' % str(suggestions))
        if len(suggestions) > 0:
            out_msg = '{"Suggestions": %s, "Error": ""}' \
                      % json.dumps([{'Id': p[0], 'Title': p[1], 'Probability': float('%.2f' % p[2]),
                                     'Repo': msg['Repository']} for p in suggestions])
            # with open('debug.txt', 'w') as f:
            #     f.write(out_msg)
        else:
            out_msg = '{"Suggestions": [], "Error": "No suggestions available"}'
        with open('previous_suggestion.txt', 'w') as f:
            f.write(out_msg)
        with open('previous_ids.txt', 'w') as f:
            f.write('%s\n%s\n%s' % (repo, pr_id, issue_id))
        send_message(out_msg)
    except Exception as e:
        # with open('debug.txt', 'w') as f:
        #     f.write('Failed Predict, %s' % str(e))
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
    out_msg = '{"Suggestions": [], "Error": "Updated model, please close and reopen plugin pop-up to use!"}'
    send_message(out_msg)


def record_links(msg):
    repo = msg['Repository'].translate({ord(c): '_' for c in '\\/'})
    local_server = xmlrpc.client.ServerProxy(SERVER_ADDR)
    for issue_id, pr_id in msg['Links']:
        local_server.record_link(repo, issue_id, pr_id)


def dummy():
    with open('previous_suggestion.txt') as f:
        out_msg = f.read()
    send_message(out_msg)


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
        try:
            if msg['Type'] == 'Prediction':
                process_prediction_request(msg)
            elif msg['Type'] == 'Update':
                model_update()
            elif msg['Type'] == 'LinkUpdate':
                record_links(msg)
            else:
                unknown_type(msg)
        except Exception as e:
            out_msg = '{"Suggestions": [], "Error": "Failed due to %s!"}' % json.dumps(e)
            send_message(out_msg)
            continue


def main():
    read_thread_func()
    sys.exit(0)


if __name__ == '__main__':
    main()
