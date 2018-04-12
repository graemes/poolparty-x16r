#!/bin/bash

while true
do
	echo "Pres CTRL+C to stop..."
	./poolparty -a x16r -o stratum+tcp://cryptopool.party:3636 -u RH4KkDFJV7FuURwrZDyQZoPWAKc4hSHuDU -p x --donate=1
done
