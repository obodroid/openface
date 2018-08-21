#!/bin/bash
#
# Download OpenFace models.

cd "$(dirname "$0")"

die() {
  echo >&2 $*
  exit 1
}

checkCmd() {
  command -v $1 >/dev/null 2>&1 \
    || die "'$1' command not found. Please install from your package manager."
}

checkCmd wget
checkCmd bunzip2

mkdir -p dlib
if [ ! -f dlib/shape_predictor_68_face_landmarks.dat ]; then
  printf "\n\n====================================================\n"
  printf "Downloading dlib's public domain face landmarks model.\n"
  printf "Reference: https://github.com/davisking/dlib-models\n\n"
  printf "====================================================\n\n"
  wget -nv \
       http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
       -O dlib/shape_predictor_68_face_landmarks.dat.bz2
  [ $? -eq 0 ] || die "+ Error in wget."
  bunzip2 dlib/shape_predictor_68_face_landmarks.dat.bz2
  [ $? -eq 0 ] || die "+ Error using bunzip2."

  wget -nv \
       http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 \
       -O dlib/shape_predictor_5_face_landmarks.dat.bz2
  [ $? -eq 0 ] || die "+ Error in wget."
  bunzip2 dlib/shape_predictor_5_face_landmarks.dat.bz2
  [ $? -eq 0 ] || die "+ Error using bunzip2."

  wget -nv \
       http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 \
       -O dlib/dlib_face_recognition_resnet_model_v1.dat.bz2
  [ $? -eq 0 ] || die "+ Error in wget."
  bunzip2 dlib/dlib_face_recognition_resnet_model_v1.dat.bz2
  [ $? -eq 0 ] || die "+ Error using bunzip2."
fi

mkdir -p openface
if [ ! -f openface/nn4.small2.v1.t7 ]; then
  printf "\n\n====================================================\n"
  printf "Downloading OpenFace models, which are copyright\n"
  printf "Carnegie Mellon University and are licensed under\n"
  printf "the Apache 2.0 License.\n\n"
  printf "====================================================\n\n"

  wget -nv \
       https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7 \
       -O openface/nn4.small2.v1.t7
  [ $? -eq 0 ] || ( rm openface/nn4.small2.v1.t7* && die "+ nn4.small2.v1.t7: Error in wget." )

  wget -nv \
       https://storage.cmusatyalab.org/openface-models/celeb-classifier.nn4.small2.v1.pkl \
       -O openface/celeb-classifier.nn4.small2.v1.pkl
  [ $? -eq 0 ] || ( rm openface/celeb-classifier.nn4.small2.v1.pkl && \
                    die "+ celeb-classifier.nn4.small2.v1.pkl: Error in wget." )
fi

printf "\n\n====================================================\n"
printf "Verifying checksums.\n"
printf "====================================================\n\n"

md5str() {
  local FNAME=$1
  case $(uname) in
    "Linux")
      echo $(md5sum "$FNAME" | cut -d ' ' -f 1)
      ;;
    "Darwin")
      echo $(md5 -q "$FNAME")
      ;;
  esac
}

checkmd5() {
  local FNAME=$1
  local EXPECTED=$2
  local ACTUAL=$(md5str "$FNAME")
  if [ $EXPECTED = $ACTUAL ]; then
    printf "+ $FNAME: successfully checked\n"
  else
    printf "+ ERROR! $FNAME md5sum did not match.\n"
    printf "  + Expected: $EXPECTED\n"
    printf "  + Actual: $ACTUAL\n"
    printf "  + Please manually delete this file and try re-running this script.\n"
    return -1
  fi
  printf "\n"
}

set -e

checkmd5 \
  dlib/shape_predictor_68_face_landmarks.dat \
  73fde5e05226548677a050913eed4e04

checkmd5 \
  dlib/shape_predictor_5_face_landmarks.dat \
  79db4b6294389d5ec3923c422efad7ec

checkmd5 \
  dlib/dlib_face_recognition_resnet_model_v1.dat \
  2316b25ae80acf4ad9b620b00071c423

checkmd5 \
  openface/celeb-classifier.nn4.small2.v1.pkl \
  199a2c0d32fd0f22f14ad2d248280475

checkmd5 \
  openface/nn4.small2.v1.t7 \
  c95bfd8cc1adf05210e979ff623013b6
