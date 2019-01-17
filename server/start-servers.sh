#!/bin/bash

set -e -u

function die { echo $1; exit 42; }

cd $(dirname $0)
trap 'kill $(jobs -p)' EXIT

cat <<EOF

Starting the HTTP TLS server on port HTTP_PORT
and the Secure WebSocket server on port WEBSOCKET_PORT.

Access the demo through the HTTP server in your browser.
If you're running on the same computer outside of Docker, use https://localhost:HTTP_PORT
If you're running on the same computer with Docker, find the IP
address of the Docker container and use https://<docker-ip>:HTTP_PORT.
If you're running on a remote computer, find the IP address
and use https://<remote-ip>:HTTP_PORT.

WARNING: Chromium will warn on self-signed certificates. Please accept the certificate
and reload the app.

EOF

WEBSOCKET_LOG='/tmp/openface.websocket.log'
printf "WebSocket Server: Logging to '%s'\n\n" $WEBSOCKET_LOG

python2 simpleSSLServer.py &> /dev/null &
./websocket-server.py 2>&1 | tee $WEBSOCKET_LOG &

wait
