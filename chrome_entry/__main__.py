import json
import struct
import sys
import xmlrpc.client


# Helper function that sends a message to the webapp.
def send_message(message):
    # Write message size.
    sys.stdout.buffer.write(struct.pack('i', len(message)))
    # Write the message itself.
    sys.stdout.buffer.write(message.encode('utf-8'))
    sys.stdout.buffer.flush()


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
        repo = msg['Repository'].translate({ord(c): '_' for c in '\\/'})
        pr_id = msg['PR']
        issue_id = msg['Issue']
        out_msg = '{"Suggestions": [], "Error": "Received data, loading model."}'
        send_message(out_msg)

        try:
            out_msg = '{"Suggestions": [], "Error": "Model loaded, running predictions."}'
            send_message(out_msg)
            local_server = xmlrpc.client.ServerProxy('http://localhost:8000')
            if pr_id:
                suggestions = local_server.predict_pr(repo, pr_id)
            elif issue_id:
                suggestions = local_server.predict_issue(repo, issue_id)
            # with open('debug.txt', 'w') as f:
            #     f.write('Got suggestions: %s' % str(suggestions))
            if len(suggestions) > 0:
                out_msg = '{"Suggestions": %s, "Error": ""}' \
                          % json.dumps([{'Id': p[0][len('issue_'):], 'Probability': float('%.2f' % p[1]),
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
                out_msg = '{"Suggestions": [], "Error": "Failed to generate suggestions due to %s"}' % str(e)
            send_message(out_msg)


def main():
    read_thread_func()
    sys.exit(0)


if __name__ == '__main__':
    main()
