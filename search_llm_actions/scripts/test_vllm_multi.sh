#!/bin/bash

if [ ! -f "server_ports.txt" ]; then
    echo "Error: server_ports.txt file not found"
    exit 1
fi


IFS=' ' read -r -a SERVER_PORTS < server_ports.txt
if [ ${#SERVER_PORTS[@]} -eq 0 ]; then
    echo "Error: No ports found in server_ports.txt"
    exit 1
fi
echo "Loaded ports: ${SERVER_PORTS[*]}"


for SERVER_PORT in ${SERVER_PORTS[@]}; do
    echo -n "Checking port ${SERVER_PORT}: "
    if curl -s -f -m 5 http://localhost:${SERVER_PORT}/v1/models > /dev/null; then
        echo "succeeded"
    else
        echo "failed"
    fi
done
