#!/bin/bash
# Run Remote
set -euo pipefail
REMOTEHOST="$1"
CMD="${@:2}"
echo "running on $REMOTEHOST cmd: $CMD"

MD5=$(tar --exclude='.git' --exclude='.md5sum' --exclude='rr.sh' -cf - . | md5)
touch .md5sum
if [ $MD5 == $(cat .md5sum) ]; then
	echo nothing changed
else
	echo change, syncing
	echo $MD5 >.md5sum
	rsync -av -e ssh --exclude '.git' . $REMOTEHOST:~/repos/jem-experiments
fi

SETUP_PYENV="PATH=\$HOME/.pyenv/bin:\$PATH; eval \"\$(pyenv init -)\""
SETUP_REQS="cd repos/jem-experiments; pip install -r requirements.txt"

ssh $REMOTEHOST "$SETUP_PYENV; $SETUP_REQS; $CMD"
