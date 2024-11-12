#!/usr/bin/env bash

basedir=$(dirname "$0")
[[ -z "$GPUID" ]] && GPUID=0

[[ ! -d "$basedir/programs/feat" ]] && mkdir -p "$basedir/programs/feat"


datadir="$basedir/../../dataset"
if [[ ! -d "$datadir" ]]; then
  echo "You must download the dataset first!"
  exit 1
fi

echo "Copying ground truth instructions"
cat "$datadir/test_real.txt" | while read -r name; do
  echo "$name"
  cp "$datadir/real/160x160/rgb/${name}.jpg" "$basedir/programs/feat/"
  tput cuu1 && tput el
done
echo "Done"


# #!/usr/bin/env bash

# basedir=$(dirname "$0")
# [[ -z "$GPUID" ]] && GPUID=0

# [[ ! -d "$basedir/programs/test" ]] && mkdir -p "$basedir/programs/test"


# datadir="$basedir/../../dataset"
# if [[ ! -d "$datadir" ]]; then
#   echo "You must download the dataset first!"
#   exit 1
# fi

# echo "Copying ground truth instructions"
# cat "$datadir/test_real.txt" | while read -r name; do
#   echo "$name"
#   cp "$datadir/real/160x160/rgb/${name}.jpg" "$basedir/programs/test/"
#   tput cuu1 && tput el
# done
# echo "Done"




